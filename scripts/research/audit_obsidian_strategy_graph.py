#!/usr/bin/env python3
"""Read-only integrity audit for the LuminaQuant Obsidian strategy graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        return {}
    parsed = yaml.safe_load(text[4:].split("\n---\n", 1)[0])
    return parsed if type(parsed) is dict else {}


def canonical_target(raw: str) -> str:
    return raw.split("|", 1)[0].split("#", 1)[0].strip()


def resolve_target(
    target: str,
    *,
    vault: Path,
    paths_by_stem: dict[str, list[Path]],
) -> list[Path]:
    if not target:
        return []
    if "/" in target:
        candidate = vault / target
        if candidate.suffix.lower() != ".md":
            candidate = Path(f"{candidate}.md")
        return [candidate] if candidate.is_file() else []
    return paths_by_stem.get(target, [])


def verify_generated_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "_generated_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files") or manifest.get("entries") or manifest.get("artifacts")
    if type(entries) is not list:
        raise ValueError("generated_manifest_entries_invalid")
    missing: list[str] = []
    mismatched: list[str] = []
    for row in entries:
        if type(row) is not dict:
            raise ValueError("generated_manifest_row_invalid")
        relative = row.get("path") or row.get("relative_path")
        expected = row.get("sha256")
        if type(relative) is not str or type(expected) is not str:
            raise ValueError("generated_manifest_binding_invalid")
        path = root / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256(path) != expected:
            mismatched.append(relative)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "entry_count": len(entries),
        "missing": missing,
        "mismatched": mismatched,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--namespace", default="LuminaQuant")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    vault = args.vault.resolve(strict=True)
    namespace = (vault / args.namespace).resolve(strict=True)
    if namespace.parent != vault or not namespace.is_dir():
        raise ValueError("namespace must name one direct vault directory")
    all_notes = sorted(vault.rglob("*.md"))
    resolvable = [path for path in all_notes if ".luminaquant-generated-backups" not in path.parts]
    authoritative = [
        path
        for path in namespace.rglob("*.md")
        if ".luminaquant-generated-backups" not in path.parts
    ]
    ids: dict[str, list[str]] = defaultdict(list)
    missing_ids: list[str] = []
    by_stem: dict[str, list[Path]] = defaultdict(list)
    path_casefold: dict[str, list[str]] = defaultdict(list)
    for path in resolvable:
        by_stem[path.stem].append(path)
    for path in authoritative:
        relative = path.relative_to(vault).as_posix()
        path_casefold[relative.casefold()].append(relative)
        note_id = frontmatter(path).get("id")
        if type(note_id) is str and note_id:
            ids[note_id].append(relative)
        else:
            missing_ids.append(relative)
    broken: list[dict[str, str]] = []
    ambiguous: list[dict[str, Any]] = []
    link_count = 0
    for path in authoritative:
        relative = path.relative_to(vault).as_posix()
        for raw in WIKILINK.findall(path.read_text(encoding="utf-8")):
            link_count += 1
            target = canonical_target(raw)
            resolved = resolve_target(target, vault=vault, paths_by_stem=by_stem)
            if not resolved:
                broken.append({"source": relative, "target": target})
            elif "/" not in target and len(resolved) != 1:
                ambiguous.append(
                    {
                        "source": relative,
                        "target": target,
                        "matches": [
                            candidate.relative_to(vault).as_posix() for candidate in resolved
                        ],
                    }
                )
    strategy_root = namespace / "Strategies"
    canonical_strategies = sorted(strategy_root.glob("*.md"))
    numeric_without_evaluation: list[str] = []
    rejected_without_decision: list[str] = []
    managed_section_missing: list[str] = []
    for path in canonical_strategies:
        fm = frontmatter(path)
        body = path.read_text(encoding="utf-8")
        if re.search(
            r"(?:return|sharpe|mdd|turnover|trade count):?\s*`?-?\d", body, re.I
        ) and not fm.get("evaluated_by"):
            numeric_without_evaluation.append(path.name)
        if fm.get("research_state") == "rejected" and not fm.get("rejected_by"):
            rejected_without_decision.append(path.name)
        if "<!-- alpha-research:begin -->" not in body:
            managed_section_missing.append(path.name)
    relationship_path = (
        Path(__file__).resolve().parents[2] / "docs/research_note/strategy_relationships.json"
    )
    relationships = json.loads(relationship_path.read_text(encoding="utf-8"))
    node_ids = {row["id"] for row in relationships["nodes"]}
    unresolved_edges = [
        edge
        for edge in relationships["edges"]
        if edge["source_id"] not in node_ids or edge["target_id"] not in node_ids
    ]
    graph_path = vault / ".obsidian" / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    generated_root = namespace / "Strategy Research Generated"
    generated_manifest = verify_generated_manifest(generated_root)
    receipt = {
        "artifact_kind": "luminaquant_obsidian_strategy_graph_audit.v1",
        "vault": str(vault),
        "namespace": str(namespace.relative_to(vault)),
        "order_routing_enabled": False,
        "total_markdown_notes": len(all_notes),
        "authoritative_note_count": len(authoritative),
        "authoritative_id_count": len(ids),
        "missing_ids": missing_ids,
        "duplicate_ids": {key: value for key, value in ids.items() if len(value) > 1},
        "wikilink_count": link_count,
        "broken_wikilinks": broken,
        "ambiguous_basename_wikilinks": ambiguous,
        "casefold_path_collisions": [
            values for values in path_casefold.values() if len(values) > 1
        ],
        "strategy_graph": {
            "canonical_strategy_count": len(canonical_strategies),
            "generated_strategy_count": len(list((generated_root / "Strategies").glob("*.md"))),
            "family_count": len(list((generated_root / "Families").glob("*.md"))),
            "evidence_count": len(list((generated_root / "Evidence").glob("*.md"))),
            "numeric_without_evaluation": numeric_without_evaluation,
            "rejected_without_decision": rejected_without_decision,
            "managed_section_missing": managed_section_missing,
            "relationship_node_count": len(node_ids),
            "relationship_edge_count": len(relationships["edges"]),
            "unresolved_relationship_edges": unresolved_edges,
        },
        "generated_manifest": generated_manifest,
        "graph_settings": {
            "sha256": sha256(graph_path),
            "showArrow": graph.get("showArrow"),
            "hideUnresolved": graph.get("hideUnresolved"),
            "search": graph.get("search"),
            "color_group_count": len(graph.get("colorGroups") or []),
        },
    }
    failures = {
        "missing_ids": bool(missing_ids),
        "duplicate_ids": bool(receipt["duplicate_ids"]),
        "broken_wikilinks": bool(broken),
        "ambiguous_basename_wikilinks": bool(ambiguous),
        "casefold_path_collisions": bool(receipt["casefold_path_collisions"]),
        "generated_manifest": bool(
            generated_manifest["missing"] or generated_manifest["mismatched"]
        ),
        "strategy_evidence": bool(
            numeric_without_evaluation
            or rejected_without_decision
            or managed_section_missing
            or unresolved_edges
        ),
        "graph_settings": not (
            graph.get("showArrow") is True
            and graph.get("hideUnresolved") is True
            and len(graph.get("colorGroups") or []) >= 5
        ),
    }
    receipt["failures"] = failures
    receipt["status"] = "passed" if not any(failures.values()) else "failed"
    payload = (
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "authoritative_notes": len(authoritative),
                "wikilinks": link_count,
                "broken": len(broken),
                "ambiguous": len(ambiguous),
                "output": str(args.output),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
