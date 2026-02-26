"""Audit hardcoded numeric parameters with AST-based scanning.

Detects:
- non-exempt numeric literals in Python code
- non-exempt numeric constants embedded inside formula-like strings

Outputs JSON + Markdown reports and can enforce via a baseline file.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LITERAL_PATHS: tuple[str, ...] = ("lumina_quant/strategies",)
DEFAULT_FORMULA_PATHS: tuple[str, ...] = ("lumina_quant/indicators/formulaic_definitions.py",)
DEFAULT_BASELINE = ".github/hardcoded_params_baseline.json"
DEFAULT_JSON_REPORT = "reports/quality/hardcoded_params_report.json"
DEFAULT_MD_REPORT = "reports/quality/hardcoded_params_report.md"

_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9_]+")

_EPS_EXEMPTIONS = (1e-12, 1e-9, 1e-6)


@dataclass(frozen=True)
class Violation:
    kind: str
    path: str
    line: int
    col: int
    value: float
    literal: str
    suggestion: str
    context: str

    def signature(self) -> str:
        if self.kind == "formula_string":
            return f"{self.kind}:{self.path}:{self.suggestion}"
        return f"{self.kind}:{self.path}:{self.line}:{self.col}:{self.literal}"

    def legacy_signature(self) -> str:
        return f"{self.kind}:{self.path}:{self.line}:{self.col}:{self.literal}"


@dataclass(frozen=True)
class FileScanConfig:
    path: Path
    scan_literals: bool
    scan_formula_strings: bool


def _is_power_of_two(value: float) -> bool:
    if value <= 0.0:
        return False
    mantissa, _ = math.frexp(value)
    return math.isclose(mantissa, 0.5, rel_tol=0.0, abs_tol=1e-15)


def is_exempt_value(value: float) -> bool:
    abs_value = abs(float(value))
    if math.isnan(abs_value) or math.isinf(abs_value):
        return True
    if math.isclose(abs_value, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return True
    if math.isclose(abs_value, 1.0, rel_tol=0.0, abs_tol=1e-15):
        return True
    if math.isclose(abs_value, 0.5, rel_tol=0.0, abs_tol=1e-15):
        return True
    if any(math.isclose(abs_value, eps, rel_tol=0.0, abs_tol=1e-18) for eps in _EPS_EXEMPTIONS):
        return True
    # powers-of-two family (2^k) up to 256
    return abs_value <= 256.0 and _is_power_of_two(abs_value)


def _looks_like_formula_string(text: str) -> bool:
    if not text or not any(ch.isdigit() for ch in text):
        return False
    if not any(ch in text for ch in ("+", "-", "*", "/", "^", "?", "(", ")")):
        return False
    lower = text.lower()
    formula_tokens = (
        "rank",
        "delta",
        "ts_",
        "correlation",
        "covariance",
        "sum(",
        "adv",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "vwap",
    )
    return any(token in lower for token in formula_tokens)


def _module_declares_formula_specs(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"ALPHA_FUNCTION_SPECS", "ALPHA_SPECS"}:
                    return True
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in {"ALPHA_FUNCTION_SPECS", "ALPHA_SPECS"}
        ):
            return True
    return False


@lru_cache(maxsize=512)
def _formula_is_tunable_via_ir(alpha_id: int, formula: str) -> bool:
    try:
        from lumina_quant.strategies.alpha101.formula_ir import parse_formula_to_ir

        _ = parse_formula_to_ir(int(alpha_id), str(formula))
        return True
    except Exception:
        return False


def _iter_python_files(paths: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        candidate = (PROJECT_ROOT / raw).resolve()
        if not candidate.exists():
            continue
        if candidate.is_file() and candidate.suffix == ".py":
            files.append(candidate)
            continue
        if candidate.is_dir():
            files.extend(sorted(p for p in candidate.rglob("*.py") if p.is_file()))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _build_scan_plan(literal_paths: Sequence[str], formula_paths: Sequence[str]) -> list[FileScanConfig]:
    literal_files = set(_iter_python_files(literal_paths))
    formula_files = set(_iter_python_files(formula_paths))
    plan: list[FileScanConfig] = []
    for path in sorted(literal_files | formula_files):
        plan.append(
            FileScanConfig(
                path=path,
                scan_literals=path in literal_files,
                scan_formula_strings=path in formula_files,
            )
        )
    return plan


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parent_map: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[child] = node
    return parent_map


def _value_slug(value: float) -> str:
    formatted = f"{abs(value):.12g}".replace(".", "p").replace("-", "m").replace("+", "")
    if "e" in formatted:
        formatted = formatted.replace("e", "e")
    prefix = "neg_" if value < 0 else ""
    return f"{prefix}{formatted}"


def _sanitize_token(token: str) -> str:
    parts = [part for part in _TOKEN_SPLIT.split(token.lower()) if part]
    return "_".join(parts) or "param"


def _find_enclosing_function(node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = node
    while current in parent_map:
        current = parent_map[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    return None


def _is_function_default_literal(
    node: ast.AST,
    *,
    func: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> bool:
    if func is None:
        return False
    defaults = [*list(func.args.defaults), *[item for item in func.args.kw_defaults if item is not None]]
    for default_node in defaults:
        for descendant in ast.walk(default_node):
            if descendant is node:
                return True
    return False


def _find_default_arg_name(node: ast.AST, func: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    positional_args = list(func.args.args)
    defaults = list(func.args.defaults)
    default_start = len(positional_args) - len(defaults)
    for idx, default_node in enumerate(defaults):
        if default_node is node:
            return positional_args[default_start + idx].arg
    for kwarg, default_node in zip(func.args.kwonlyargs, func.args.kw_defaults, strict=False):
        if default_node is node:
            return kwarg.arg
    return None


def _assignment_target_name(parent: ast.AST | None) -> str | None:
    if isinstance(parent, ast.Assign) and parent.targets:
        target = parent.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    if isinstance(parent, ast.AnnAssign) and isinstance(parent.target, ast.Name):
        return parent.target.id
    return None


def _context_label(node: ast.AST, parent_map: dict[ast.AST, ast.AST], *, alpha_id: int | None = None) -> str:
    labels: list[str] = []
    func = _find_enclosing_function(node, parent_map)
    if func is not None:
        labels.append(f"function:{func.name}")
    current = node
    while current in parent_map:
        current = parent_map[current]
        if isinstance(current, ast.ClassDef):
            labels.append(f"class:{current.name}")
            break
    if alpha_id is not None:
        labels.append(f"alpha:{alpha_id}")
    return " | ".join(labels) if labels else "module"


def _suggest_literal_key(
    node: ast.AST,
    parent_map: dict[ast.AST, ast.AST],
    *,
    path: Path,
    value: float,
) -> str:
    func = _find_enclosing_function(node, parent_map)
    parent = parent_map.get(node)

    if func is not None:
        arg_name = _find_default_arg_name(node, func)
        if arg_name:
            base = f"{func.name}_{arg_name}"
            return f"{_sanitize_token(base)}_{_value_slug(value)}"

    assign_name = _assignment_target_name(parent)
    if assign_name:
        return f"{_sanitize_token(assign_name)}_{_value_slug(value)}"

    if isinstance(parent, ast.keyword) and parent.arg:
        func_name = func.name if func is not None else path.stem
        return f"{_sanitize_token(f'{func_name}_{parent.arg}')}_{_value_slug(value)}"

    if func is not None:
        return f"{_sanitize_token(func.name + '_param')}_{_value_slug(value)}"

    return f"{_sanitize_token(path.stem + '_param')}_{_value_slug(value)}"


def _suggest_formula_key(
    *,
    node: ast.AST,
    parent_map: dict[ast.AST, ast.AST],
    path: Path,
    value: float,
    constant_index: int,
) -> str:
    alpha_id: int | None = None
    parent = parent_map.get(node)
    if isinstance(parent, ast.Dict):
        try:
            idx = parent.values.index(node)
        except ValueError:
            idx = -1
        if idx >= 0 and idx < len(parent.keys):
            key_node = parent.keys[idx]
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, int):
                alpha_id = int(key_node.value)

    if alpha_id is not None:
        base = f"alpha_{alpha_id}_const_{constant_index + 1}"
    else:
        func = _find_enclosing_function(node, parent_map)
        if func is not None:
            base = f"{func.name}_formula_const_{constant_index + 1}"
        else:
            base = f"{path.stem}_formula_const_{constant_index + 1}"

    return f"{_sanitize_token(base)}_{_value_slug(value)}"


def _parse_numeric_literal_from_node(
    node: ast.AST,
    parent_map: dict[ast.AST, ast.AST],
    source: str,
) -> tuple[float, str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        parent = parent_map.get(node)
        if isinstance(parent, ast.UnaryOp) and parent.operand is node and isinstance(parent.op, (ast.USub, ast.UAdd)):
            return None
        value = float(node.value)
        literal = ast.get_source_segment(source, node) or str(node.value)
        return value, literal

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = node.operand
        if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)) and not isinstance(operand.value, bool):
            base = float(operand.value)
            value = -base if isinstance(node.op, ast.USub) else base
            literal = ast.get_source_segment(source, node) or str(value)
            return value, literal

    return None


def _scan_literal_violations(
    *,
    tree: ast.AST,
    parent_map: dict[ast.AST, ast.AST],
    source: str,
    rel_path: Path,
) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        parsed = _parse_numeric_literal_from_node(node, parent_map, source)
        if parsed is None:
            continue
        value, literal = parsed
        if is_exempt_value(value):
            continue

        func = _find_enclosing_function(node, parent_map)
        if func is not None:
            if func.name == "get_param_schema":
                continue
            if func.name == "__init__" and _is_function_default_literal(node, func=func):
                continue

        parent = parent_map.get(node)
        if isinstance(parent, ast.Dict) and node in parent.keys:
            # IDs / lookup keys are not tunable parameters.
            continue

        line = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        suggestion = _suggest_literal_key(node, parent_map, path=rel_path, value=value)
        violations.append(
            Violation(
                kind="literal",
                path=rel_path.as_posix(),
                line=int(line),
                col=int(col),
                value=value,
                literal=str(literal).strip(),
                suggestion=suggestion,
                context=_context_label(node, parent_map),
            )
        )
    return violations


def _scan_formula_string_violations(
    *,
    tree: ast.AST,
    parent_map: dict[ast.AST, ast.AST],
    rel_path: Path,
    skip_tunable_alpha_formula_strings: bool = False,
) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        text = node.value
        if not _looks_like_formula_string(text):
            continue

        parent = parent_map.get(node)
        alpha_id: int | None = None
        if isinstance(parent, ast.Dict):
            try:
                idx = parent.values.index(node)
            except ValueError:
                idx = -1
            if (
                idx >= 0
                and idx < len(parent.keys)
                and isinstance(parent.keys[idx], ast.Constant)
                and isinstance(parent.keys[idx].value, int)
            ):
                alpha_id = int(parent.keys[idx].value)

        if (
            skip_tunable_alpha_formula_strings
            and alpha_id is not None
            and _formula_is_tunable_via_ir(alpha_id, text)
        ):
            continue

        for constant_index, match in enumerate(_NUMBER_PATTERN.finditer(text)):
            token = match.group(0)
            try:
                value = float(token)
            except ValueError:
                continue
            if is_exempt_value(value):
                continue

            line_offset = text.count("\n", 0, match.start())
            if line_offset == 0:
                line = int(getattr(node, "lineno", 1))
                col = int(getattr(node, "col_offset", 0)) + int(match.start())
            else:
                line = int(getattr(node, "lineno", 1)) + line_offset
                last_nl = text.rfind("\n", 0, match.start())
                col = int(match.start() - last_nl - 1)

            suggestion = _suggest_formula_key(
                node=node,
                parent_map=parent_map,
                path=rel_path,
                value=value,
                constant_index=constant_index,
            )
            context = _context_label(node, parent_map, alpha_id=alpha_id)
            violations.append(
                Violation(
                    kind="formula_string",
                    path=rel_path.as_posix(),
                    line=line,
                    col=col,
                    value=value,
                    literal=token,
                    suggestion=suggestion,
                    context=context,
                )
            )
    return violations


def collect_violations(plan: Sequence[FileScanConfig]) -> list[Violation]:
    violations: list[Violation] = []
    for entry in plan:
        source = entry.path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=entry.path.as_posix())
        parent_map = _build_parent_map(tree)
        rel_path = entry.path.relative_to(PROJECT_ROOT)
        skip_tunable_formula_strings = bool(
            entry.scan_formula_strings and _module_declares_formula_specs(tree)
        )

        if entry.scan_literals:
            violations.extend(
                _scan_literal_violations(
                    tree=tree,
                    parent_map=parent_map,
                    source=source,
                    rel_path=rel_path,
                )
            )

        if entry.scan_formula_strings:
            violations.extend(
                _scan_formula_string_violations(
                    tree=tree,
                    parent_map=parent_map,
                    rel_path=rel_path,
                    skip_tunable_alpha_formula_strings=skip_tunable_formula_strings,
                )
            )
    violations.sort(key=lambda item: (item.path, item.line, item.col, item.kind, item.literal))
    return violations


def _load_baseline_signatures(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("signatures", []) if isinstance(payload, dict) else []
    return {str(item) for item in raw}


def _write_baseline(path: Path, signatures: Iterable[str], *, literal_paths: Sequence[str], formula_paths: Sequence[str]) -> None:
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "literal_paths": list(literal_paths),
        "formula_paths": list(formula_paths),
        "signatures": sorted(set(signatures)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _is_baselined(violation: Violation, baseline_signatures: set[str]) -> bool:
    return (
        violation.signature() in baseline_signatures
        or violation.legacy_signature() in baseline_signatures
    )


def _write_json_report(
    path: Path,
    *,
    violations: Sequence[Violation],
    baselined: set[str],
    literal_paths: Sequence[str],
    formula_paths: Sequence[str],
) -> None:
    rows: list[dict[str, object]] = []
    new_count = 0
    for violation in violations:
        signature = violation.signature()
        status = "baselined" if _is_baselined(violation, baselined) else "new"
        if status == "new":
            new_count += 1
        row = asdict(violation)
        row["signature"] = signature
        row["status"] = status
        rows.append(row)

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "literal_paths": list(literal_paths),
        "formula_paths": list(formula_paths),
        "summary": {
            "total_violations": len(violations),
            "new_violations": new_count,
            "baselined_violations": len(violations) - new_count,
        },
        "violations": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_markdown_report(path: Path, *, violations: Sequence[Violation], baselined: set[str]) -> None:
    lines: list[str] = []
    now = datetime.now(UTC).isoformat()
    total = len(violations)
    new_total = sum(1 for violation in violations if not _is_baselined(violation, baselined))
    baselined_total = total - new_total

    lines.append("# Hardcoded Parameter Audit Report")
    lines.append("")
    lines.append(f"Generated: `{now}`")
    lines.append("")
    lines.append(f"- Total violations: **{total}**")
    lines.append(f"- New violations: **{new_total}**")
    lines.append(f"- Baselined violations: **{baselined_total}**")
    lines.append("")

    if not violations:
        lines.append("No violations found.")
    else:
        lines.append("| status | kind | file | line | literal | suggested_param_key | context |")
        lines.append("|---|---|---|---:|---|---|---|")
        for violation in violations:
            status = "baselined" if _is_baselined(violation, baselined) else "new"
            lines.append(
                "| {status} | {kind} | `{path}` | {line} | `{literal}` | `{suggestion}` | {context} |".format(
                    status=status,
                    kind=violation.kind,
                    path=violation.path,
                    line=violation.line,
                    literal=violation.literal.replace("|", "\\|"),
                    suggestion=violation.suggestion,
                    context=violation.context.replace("|", "\\|"),
                )
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit hardcoded numeric parameters")
    parser.add_argument(
        "--literal-paths",
        nargs="+",
        default=list(DEFAULT_LITERAL_PATHS),
        help="Files/directories for numeric literal scanning.",
    )
    parser.add_argument(
        "--formula-paths",
        nargs="+",
        default=list(DEFAULT_FORMULA_PATHS),
        help="Files/directories for formula-string numeric scanning.",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help="Baseline JSON with allowed signatures.",
    )
    parser.add_argument(
        "--json-out",
        default=DEFAULT_JSON_REPORT,
        help="JSON report output path.",
    )
    parser.add_argument(
        "--md-out",
        default=DEFAULT_MD_REPORT,
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write/update the baseline using current violations and exit success.",
    )
    parser.add_argument(
        "--fail-on-baselined",
        action="store_true",
        help="Treat baselined violations as failing too.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    literal_paths = [str(item) for item in args.literal_paths]
    formula_paths = [str(item) for item in args.formula_paths]

    plan = _build_scan_plan(literal_paths=literal_paths, formula_paths=formula_paths)
    violations = collect_violations(plan)

    baseline_path = (PROJECT_ROOT / str(args.baseline)).resolve()
    baseline_signatures = _load_baseline_signatures(baseline_path)

    if args.write_baseline:
        _write_baseline(
            baseline_path,
            (violation.signature() for violation in violations),
            literal_paths=literal_paths,
            formula_paths=formula_paths,
        )
        print(f"Baseline written: {baseline_path.relative_to(PROJECT_ROOT)} ({len(violations)} signatures)")
        return 0

    json_out = (PROJECT_ROOT / str(args.json_out)).resolve()
    md_out = (PROJECT_ROOT / str(args.md_out)).resolve()

    _write_json_report(
        json_out,
        violations=violations,
        baselined=baseline_signatures,
        literal_paths=literal_paths,
        formula_paths=formula_paths,
    )
    _write_markdown_report(md_out, violations=violations, baselined=baseline_signatures)

    new_violations = [
        violation for violation in violations if not _is_baselined(violation, baseline_signatures)
    ]

    print(
        "Hardcoded parameter audit: "
        f"total={len(violations)} new={len(new_violations)} baselined={len(violations) - len(new_violations)}"
    )
    print(f"JSON report: {json_out.relative_to(PROJECT_ROOT)}")
    print(f"Markdown report: {md_out.relative_to(PROJECT_ROOT)}")

    if args.fail_on_baselined:
        return 1 if violations else 0
    return 1 if new_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
