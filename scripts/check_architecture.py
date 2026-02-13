"""Static architecture checks for modular package boundaries."""

from __future__ import annotations

import ast
import pathlib
from collections import defaultdict

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = PROJECT_ROOT / "lumina_quant"

LAYER_RULES = {
    "lumina_quant.core": {
        "deny_prefixes": (
            "lumina_quant.backtesting",
            "lumina_quant.live",
            "lumina_quant.optimization",
            "lumina_quant.infra",
        )
    },
    "lumina_quant.backtesting": {"deny_prefixes": ("lumina_quant.live",)},
}


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).with_suffix("")
    return ".".join(rel.parts)


def _iter_python_files() -> list[pathlib.Path]:
    return sorted(PKG_ROOT.rglob("*.py"))


def _extract_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _layer_for_module(module: str) -> str | None:
    for layer in LAYER_RULES:
        if module.startswith(layer):
            return layer
    return None


def _check_layer_violations() -> list[str]:
    violations: list[str] = []
    for file_path in _iter_python_files():
        module = _module_name(file_path)
        layer = _layer_for_module(module)
        if layer is None:
            continue
        imports = _extract_imports(file_path)
        deny_prefixes = LAYER_RULES[layer]["deny_prefixes"]
        for imported in imports:
            if any(imported.startswith(prefix) for prefix in deny_prefixes):
                violations.append(f"{module} imports forbidden dependency {imported}")
    return violations


def _build_local_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    all_modules = {_module_name(path) for path in _iter_python_files()}
    for file_path in _iter_python_files():
        module = _module_name(file_path)
        for imported in _extract_imports(file_path):
            if imported in all_modules:
                graph[module].add(imported)
    return graph


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    color: dict[str, int] = {}
    stack: list[str] = []
    cycles: list[list[str]] = []

    def dfs(node: str) -> None:
        color[node] = 1
        stack.append(node)
        for nxt in graph.get(node, set()):
            state = color.get(nxt, 0)
            if state == 0:
                dfs(nxt)
            elif state == 1:
                cycle_start = stack.index(nxt)
                cycles.append([*stack[cycle_start:], nxt])
        color[node] = 2
        stack.pop()

    for node in graph:
        if color.get(node, 0) == 0:
            dfs(node)
    return cycles


def main() -> int:
    violations = _check_layer_violations()
    cycles = _find_cycles(_build_local_graph())
    if not violations and not cycles:
        print("Architecture check passed.")
        return 0

    for item in violations:
        print(f"[LAYER VIOLATION] {item}")
    for cycle in cycles:
        print(f"[CYCLE] {' -> '.join(cycle)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
