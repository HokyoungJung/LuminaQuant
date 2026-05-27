#!/usr/bin/env python3
"""Remove local generated artifacts while preserving research evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CleanupCandidate:
    """A generated local path that is safe to remove."""

    path: Path
    reason: str

    @property
    def relative_path(self) -> str:
        return self.path.as_posix()


@dataclass(frozen=True)
class CleanupSummary:
    """Cleanup result summary."""

    repo_root: str
    applied: bool
    candidate_count: int
    removed_count: int
    missing_count: int
    preserved_roots: tuple[str, ...]
    removed: tuple[str, ...]
    candidates: tuple[str, ...]


DEFAULT_TARGETS: tuple[tuple[str, str], ...] = (
    (".pytest_cache", "pytest cache"),
    (".ruff_cache", "ruff cache"),
    (".mypy_cache", "mypy cache"),
    (".pyre", "pyre cache"),
    (".tox", "tox environment"),
    (".nox", "nox environment"),
    ("htmlcov", "coverage HTML output"),
    ("build", "Python build output"),
    ("dist", "Python distribution output"),
    ("lumina_quant.egg-info", "editable-install metadata"),
    ("src/lumina_quant.egg-info", "editable-install metadata"),
    ("logs", "root runtime logs"),
    ("reports/quality", "local quality/audit reports"),
    ("apps/dashboard_web/.next", "Next.js build output"),
    ("apps/dashboard_web/node_modules", "dashboard dependency install"),
    ("apps/dashboard_web/tsconfig.tsbuildinfo", "TypeScript incremental cache"),
    ("apps/dashboard_web/package-lock.json", "ignored dashboard install lock"),
    (".omx/cache", "OMX runtime cache"),
    (".omx/tmp", "OMX runtime temp files"),
    (".omx/logs", "OMX runtime logs"),
)

OPTIONAL_NATIVE_TARGETS: tuple[tuple[str, str], ...] = (
    ("native/rust_metrics/target", "optional Rust metrics build target"),
    ("native/rust_rawfirst/target", "optional Rust raw-first build target"),
    ("native/rust_hybrid_optuna/target", "optional Rust Optuna hybrid build target"),
    ("native/rust_live_signals/target", "optional Rust live-signal build target"),
)

DEFAULT_PATTERNS: tuple[tuple[str, str], ...] = (
    (".coverage", "coverage data"),
    (".coverage.*", "coverage data"),
)

PRESERVED_ROOTS: tuple[Path, ...] = (
    Path(".git"),
    Path(".env"),
    Path(".venv"),
    Path("data"),
    Path("var/reports"),
    Path("var/logs"),
    Path("docs"),
    Path("best_optimized_parameters"),
    Path(".omx/notepad.md"),
    Path(".omx/context"),
    Path(".omx/plans"),
    Path(".omx/project-memory.json"),
    Path(".omx/reports"),
    Path(".omx/goals"),
    Path(".omx/interviews"),
    Path(".omx/manual-worktrees"),
    Path("native/rust_metrics/target"),
    Path("native/rust_rawfirst/target"),
    Path("native/rust_hybrid_optuna/target"),
    Path("native/rust_live_signals/target"),
)

WALK_PRUNE_DIRS: tuple[Path, ...] = (
    *PRESERVED_ROOTS,
    Path(".gitnexus"),
    Path("apps/dashboard_web/node_modules"),
    Path("apps/dashboard_web/.next"),
)


def _normalize_repo_root(repo_root: Path) -> Path:
    root = repo_root.expanduser().resolve()
    if not (root / ".git").exists():
        raise ValueError(f"not a git repository root: {root}")
    return root


def _is_under_or_equal(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _is_preserved(rel_path: Path, preserved_roots: tuple[Path, ...] = PRESERVED_ROOTS) -> bool:
    return any(_is_under_or_equal(rel_path, preserved) for preserved in preserved_roots)


def _is_pruned(rel_path: Path) -> bool:
    return any(_is_under_or_equal(rel_path, pruned) for pruned in WALK_PRUNE_DIRS)


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _make_candidate(
    root: Path, rel_text: str, reason: str, *, allow_preserved: bool = False
) -> CleanupCandidate | None:
    rel_path = Path(rel_text)
    target = root / rel_path
    if not _path_exists(target):
        return None
    if not allow_preserved and _is_preserved(rel_path):
        return None
    return CleanupCandidate(path=rel_path, reason=reason)


def _explicit_candidates(
    root: Path,
    *,
    include_gitnexus_parse_cache: bool = False,
    include_venv: bool = False,
    include_native_targets: bool = False,
) -> list[CleanupCandidate]:
    candidates: list[CleanupCandidate] = []
    for rel_text, reason in DEFAULT_TARGETS:
        candidate = _make_candidate(root, rel_text, reason)
        if candidate is not None:
            candidates.append(candidate)

    if include_native_targets:
        for rel_text, reason in OPTIONAL_NATIVE_TARGETS:
            candidate = _make_candidate(root, rel_text, reason, allow_preserved=True)
            if candidate is not None:
                candidates.append(candidate)

    if include_gitnexus_parse_cache:
        candidate = _make_candidate(
            root,
            ".gitnexus/parse-cache",
            "GitNexus parser cache",
            allow_preserved=True,
        )
        if candidate is not None:
            candidates.append(candidate)

    if include_venv:
        candidate = _make_candidate(
            root, ".venv", "local virtual environment", allow_preserved=True
        )
        if candidate is not None:
            candidates.append(candidate)

    for pattern, reason in DEFAULT_PATTERNS:
        for path in root.glob(pattern):
            rel_path = path.relative_to(root)
            if _path_exists(path) and not _is_preserved(rel_path):
                candidates.append(CleanupCandidate(path=rel_path, reason=reason))
    return candidates


def _python_artifact_candidates(root: Path) -> list[CleanupCandidate]:
    candidates: list[CleanupCandidate] = []
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        rel_current = current_path.relative_to(root)

        kept_dirs: list[str] = []
        for dirname in dirs:
            child_rel = rel_current / dirname if rel_current != Path(".") else Path(dirname)
            if dirname == "__pycache__":
                candidates.append(CleanupCandidate(path=child_rel, reason="Python bytecode cache"))
                continue
            if _is_pruned(child_rel):
                continue
            kept_dirs.append(dirname)
        dirs[:] = kept_dirs

        if _is_pruned(rel_current):
            continue
        for filename in files:
            if filename.endswith((".pyc", ".pyo")):
                file_rel = rel_current / filename if rel_current != Path(".") else Path(filename)
                if not _is_preserved(file_rel):
                    candidates.append(
                        CleanupCandidate(path=file_rel, reason="Python bytecode file")
                    )
    return candidates


def _collapse_candidates(candidates: list[CleanupCandidate]) -> tuple[CleanupCandidate, ...]:
    by_path: dict[Path, CleanupCandidate] = {}
    for candidate in candidates:
        by_path.setdefault(candidate.path, candidate)

    selected: list[CleanupCandidate] = []
    selected_paths: set[Path] = set()
    for candidate in sorted(
        by_path.values(), key=lambda item: (len(item.path.parts), item.relative_path)
    ):
        if any(parent in selected_paths for parent in candidate.path.parents):
            continue
        selected.append(candidate)
        selected_paths.add(candidate.path)
    return tuple(selected)


def discover_candidates(
    repo_root: Path,
    *,
    include_gitnexus_parse_cache: bool = False,
    include_venv: bool = False,
    include_native_targets: bool = False,
) -> tuple[CleanupCandidate, ...]:
    """Return generated local paths eligible for cleanup."""
    root = _normalize_repo_root(repo_root)
    candidates = _explicit_candidates(
        root,
        include_gitnexus_parse_cache=include_gitnexus_parse_cache,
        include_venv=include_venv,
        include_native_targets=include_native_targets,
    )
    candidates.extend(_python_artifact_candidates(root))
    return _collapse_candidates(candidates)


def _delete_path(path: Path) -> bool:
    if not _path_exists(path):
        return False
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    if path.is_dir():
        shutil.rmtree(path)
        return True
    path.unlink(missing_ok=True)
    return True


def run_cleanup(
    repo_root: Path,
    *,
    apply: bool = False,
    include_gitnexus_parse_cache: bool = False,
    include_venv: bool = False,
    include_native_targets: bool = False,
) -> CleanupSummary:
    """Discover, and optionally remove, local generated artifacts."""
    root = _normalize_repo_root(repo_root)
    candidates = discover_candidates(
        root,
        include_gitnexus_parse_cache=include_gitnexus_parse_cache,
        include_venv=include_venv,
        include_native_targets=include_native_targets,
    )
    removed: list[str] = []
    missing_count = 0
    if apply:
        for candidate in candidates:
            if _delete_path(root / candidate.path):
                removed.append(candidate.relative_path)
            else:
                missing_count += 1

    return CleanupSummary(
        repo_root=root.as_posix(),
        applied=apply,
        candidate_count=len(candidates),
        removed_count=len(removed),
        missing_count=missing_count,
        preserved_roots=tuple(path.as_posix() for path in PRESERVED_ROOTS),
        removed=tuple(removed),
        candidates=tuple(candidate.relative_path for candidate in candidates),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clean local generated artifacts while preserving data, research reports, notes, "
            "and paper/testnet evidence. Defaults to dry-run."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--apply", action="store_true", help="delete discovered candidates")
    parser.add_argument(
        "--include-gitnexus-parse-cache",
        action="store_true",
        help="also remove .gitnexus/parse-cache; the index can be rebuilt with npx gitnexus analyze",
    )
    parser.add_argument(
        "--include-venv",
        action="store_true",
        help="also remove .venv; off by default to keep local verification cheap",
    )
    parser.add_argument(
        "--include-native-targets",
        action="store_true",
        help=(
            "also remove native Rust target directories; off by default so Python wrappers "
            "can keep loading proven release shared libraries"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_cleanup(
        args.repo_root,
        apply=args.apply,
        include_gitnexus_parse_cache=args.include_gitnexus_parse_cache,
        include_venv=args.include_venv,
        include_native_targets=args.include_native_targets,
    )
    if args.json:
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    mode = "APPLY" if summary.applied else "DRY-RUN"
    print(f"{mode}: {summary.candidate_count} cleanup candidates")
    for rel_path in summary.candidates:
        marker = "removed" if summary.applied and rel_path in summary.removed else "candidate"
        print(f"- {marker}: {rel_path}")
    if summary.applied:
        print(f"removed={summary.removed_count} missing={summary.missing_count}")
    print("preserved roots:")
    for rel_path in summary.preserved_roots:
        print(f"- {rel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
