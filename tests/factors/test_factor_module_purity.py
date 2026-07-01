"""Static purity guard for factor/indicator modules.

AST-scans every ``lumina_quant.indicators`` module and asserts the absence of
system/network access primitives (blocklist). Numeric/data libraries
(``numpy``/``pandas``/``polars``) and ``lumina_quant`` internals are allowed.

This is a read-only static check: it never imports or executes the scanned
modules, so it cannot alter any numeric behavior. Covers AC-33.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

# Top-level module names that must never be imported by factor/indicator code.
# These grant filesystem, process, or network access and have no place inside a
# deterministic numeric factor library.
BLOCKED_IMPORT_ROOTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "urllib3",
        "requests",
        "http",
        "ftplib",
        "smtplib",
        "telnetlib",
        "asyncio",
    }
)

# Builtin callables that indicate dynamic execution or raw file/network I/O.
BLOCKED_CALL_NAMES = frozenset({"eval", "exec", "open", "__import__"})


def _indicator_package_dir() -> Path:
    package = importlib.import_module("lumina_quant.indicators")
    module_file = getattr(package, "__file__", None)
    assert module_file is not None, "lumina_quant.indicators must be an importable package"
    return Path(module_file).resolve().parent


def _iter_module_files() -> list[Path]:
    root = _indicator_package_dir()
    files = sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    return files


def _import_roots(tree: ast.AST) -> list[tuple[str, int]]:
    """Return (top_level_module, lineno) for every absolute import statement."""

    roots: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                roots.append((root, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (``from . import x``) are internal and always allowed.
            if node.level and node.level > 0:
                continue
            if node.module is None:
                continue
            root = node.module.split(".", 1)[0]
            roots.append((root, node.lineno))
    return roots


def _blocked_calls(tree: ast.AST) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALL_NAMES:
                hits.append((node.func.id, node.lineno))
    return hits


def test_indicator_package_scan_is_non_trivial():
    files = _iter_module_files()
    # Guard against a vacuous pass if the package layout ever changes.
    assert len(files) >= 5, f"expected several indicator modules, found {len(files)}"


@pytest.mark.parametrize(
    "module_path",
    _iter_module_files(),
    ids=lambda path: path.name,
)
def test_indicator_module_has_no_blocked_imports(module_path: Path):
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))

    offending_imports = [
        (root, lineno)
        for root, lineno in _import_roots(tree)
        if root in BLOCKED_IMPORT_ROOTS
    ]
    assert not offending_imports, (
        f"{module_path.name} imports blocklisted modules: {offending_imports}"
    )


@pytest.mark.parametrize(
    "module_path",
    _iter_module_files(),
    ids=lambda path: path.name,
)
def test_indicator_module_has_no_dynamic_execution_calls(module_path: Path):
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))

    offending_calls = _blocked_calls(tree)
    assert not offending_calls, (
        f"{module_path.name} uses blocklisted builtin calls: {offending_calls}"
    )
