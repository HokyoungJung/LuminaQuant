"""Repository safety audit for the sanitized public export."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATH_PARTS = {
    ".env",
    ".omx",
    "best_optimized_parameters",
    "config.yaml",
    "configs",
    "data",
    "docs/research",
    "logs",
    "reports",
    "scripts/research",
    "var",
}
FORBIDDEN_TEXT = {
    "alpha_zoo",
    "artifact_portfolio",
    "binance",
    "crypto_fx",
    "hyperliquid",
    "mt5",
    "polymarket",
    "profit_moonshot",
    "quants-agent",
    "research_note",
    "state_distilled",
    "private/main",
    "private-ci",
    "api_secret",
    "secret_key",
    "access_key",
}
ALLOWED_SUFFIXES = {".py", ".rs", ".md", ".toml", ".yml", ".yaml", ".csv", ".txt", ""}


def _iter_repo_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        ignored_parts = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "target",
        }
        lock_files = {"uv.lock", "Cargo.lock"}
        if path.name in lock_files or any(part in ignored_parts for part in path.parts):
            continue
        if rel == "tests/public_safety_audit.py":
            continue
        if path.is_file():
            files.append(path)
        for forbidden in FORBIDDEN_PATH_PARTS:
            if rel == forbidden or rel.startswith(f"{forbidden}/"):
                raise AssertionError(f"forbidden public path present: {rel}")
    return files


def run() -> None:
    for path in _iter_repo_files():
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix not in ALLOWED_SUFFIXES:
            raise AssertionError(f"unexpected public file type: {rel}")
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for forbidden in FORBIDDEN_TEXT:
            if forbidden in lowered:
                raise AssertionError(f"forbidden text '{forbidden}' found in {rel}")


if __name__ == "__main__":
    run()
