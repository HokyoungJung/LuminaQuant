"""Validate README/docs markdown links and required install snippets."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs"

TARGETS = [ROOT / "README.md", ROOT / "README_KR.md", *sorted(DOCS_ROOT.rglob("*.md"))]

LINK_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)|\[[^\]]+]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:", "#")


def _pyproject_pin(package: str) -> str:
    """Return the current requirement string for ``package`` from pyproject.toml.

    Derived dynamically so this verifier can never drift from the real pins
    (hardcoded snippets here went stale after the Stack Lift version bumps).
    Environment markers (the part after ';') are stripped — READMEs document
    the version constraint, not the platform gate.
    """
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pools: list[list[str]] = [data["project"].get("dependencies", [])]
    pools.extend(data["project"].get("optional-dependencies", {}).values())
    for pool in pools:
        for req in pool:
            name = re.split(r"[><=!~;\[ ]", req.strip(), maxsplit=1)[0]
            if name == package:
                return req.split(";", 1)[0].strip()
    raise SystemExit(f"verify_docs: package {package!r} not found in pyproject.toml")


_README_SNIPPETS = [
    _pyproject_pin("polars"),
    _pyproject_pin("cudf-polars-cu12"),
    "uv sync --extra gpu",
]

REQUIRED_SNIPPETS = {
    ROOT / "README.md": _README_SNIPPETS,
    ROOT / "README_KR.md": _README_SNIPPETS,
}


def _iter_markdown_links(text: str) -> list[str]:
    links: list[str] = []
    for match in LINK_RE.finditer(text):
        target = match.group(1) or match.group(2)
        if target:
            links.append(target.strip())
    return links


def _resolve_local_path(source_file: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().strip("<>")
    if not target or target.startswith(SKIP_PREFIXES):
        return None

    target = target.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0].strip()
    if not target:
        return None

    decoded_target = unquote(target)
    resolved = (source_file.parent / decoded_target).resolve()
    root_resolved = ROOT.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        return None
    return resolved


def main() -> int:
    errors: list[str] = []

    for file_path in TARGETS:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        for raw_target in _iter_markdown_links(text):
            resolved = _resolve_local_path(file_path, raw_target)
            if resolved is None:
                continue
            if not resolved.exists():
                rel_file = file_path.relative_to(ROOT)
                rel_target = resolved.relative_to(ROOT)
                errors.append(f"{rel_file}: missing link target `{raw_target}` -> `{rel_target}`")

    for file_path, snippets in REQUIRED_SNIPPETS.items():
        text = file_path.read_text(encoding="utf-8", errors="replace")
        for snippet in snippets:
            if snippet not in text:
                rel_file = file_path.relative_to(ROOT)
                errors.append(f"{rel_file}: missing required snippet `{snippet}`")

    if errors:
        print("❌ Documentation verification failed:")
        for item in errors:
            print(f"- {item}")
        return 1

    print(f"✅ Documentation verification passed ({len(TARGETS)} markdown files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
