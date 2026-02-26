"""Alpha101 formula parsing + AST IR utilities."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass

_EXEMPT_EPSILONS = (1e-12, 1e-9, 1e-6)
_EXEMPT_ABS_VALUES = frozenset({0.0, 0.5, 1.0})
_ADV_NAME_RE = re.compile(r"^adv\d+$")

ROOT_NAMES = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "returns",
        "cap",
        "sector",
        "industry",
        "subindustry",
    }
)

CALL_NAMES = frozenset(
    {
        "abs",
        "log",
        "sign",
        "rank",
        "ts_rank",
        "ts_sum",
        "ts_stddev",
        "ts_corr",
        "ts_cov",
        "ts_min",
        "ts_max",
        "ts_product",
        "delay",
        "delta",
        "ts_argmax",
        "ts_argmin",
        "decay_linear",
        "scale",
        "signed_power",
        "where",
        "indneutralize",
        "max",
        "min",
    }
)

POLARS_CALLS = frozenset({"abs", "log", "sign", "where", "max", "min"})


@dataclass(frozen=True, slots=True)
class ConstantSlot:
    key: str
    default: float


@dataclass(frozen=True, slots=True)
class FormulaIR:
    alpha_id: int
    source: str
    normalized: str
    tree: ast.Expression
    constant_slots: dict[int, ConstantSlot]


def _is_power_of_two(value: float) -> bool:
    if value <= 0.0:
        return False
    mantissa, _ = math.frexp(value)
    return math.isclose(mantissa, 0.5, rel_tol=0.0, abs_tol=1e-15)


def is_exempt_constant(value: float) -> bool:
    abs_value = abs(float(value))
    if math.isnan(abs_value) or math.isinf(abs_value):
        return True
    if any(math.isclose(abs_value, base, rel_tol=0.0, abs_tol=1e-15) for base in _EXEMPT_ABS_VALUES):
        return True
    if any(math.isclose(abs_value, eps, rel_tol=0.0, abs_tol=1e-18) for eps in _EXEMPT_EPSILONS):
        return True
    return abs_value <= 256.0 and _is_power_of_two(abs_value)


def _convert_ternary(expr: str) -> str:
    text = str(expr)
    while "?" in text:
        q_pos = text.rfind("?")
        colon = -1
        nested = 0
        depth = 0
        for idx in range(q_pos + 1, len(text)):
            char = text[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "?" and depth >= 0:
                nested += 1
            elif char == ":" and depth >= 0:
                if nested == 0:
                    colon = idx
                    break
                nested -= 1
        if colon < 0:
            break

        left = q_pos
        paren_depth = 0
        while left >= 0:
            char = text[left]
            if char == ")":
                paren_depth += 1
            elif char == "(":
                if paren_depth == 0:
                    break
                paren_depth -= 1
            left -= 1
        if left < 0:
            break

        right = colon
        paren_depth = 0
        while right < len(text):
            char = text[right]
            if char == "(":
                paren_depth += 1
            elif char == ")":
                if paren_depth == 0:
                    break
                paren_depth -= 1
            right += 1
        if right >= len(text):
            break

        cond = text[left + 1 : q_pos].strip()
        if_true = text[q_pos + 1 : colon].strip()
        if_false = text[colon + 1 : right].strip()
        replacement = f"where({cond}, {if_true}, {if_false})"
        text = text[:left] + replacement + text[right + 1 :]
    return text.strip()


def normalize_formula(expr: str) -> str:
    code = _convert_ternary(expr)
    replacements = {
        "Ts_ArgMax": "ts_argmax",
        "Ts_ArgMin": "ts_argmin",
        "Ts_Rank": "ts_rank",
        "SignedPower": "signed_power",
        "IndNeutralize": "indneutralize",
        "indneutralize": "indneutralize",
        "IndClass.subindustry": "subindustry",
        "IndClass.industry": "industry",
        "IndClass.sector": "sector",
        "Sign": "sign",
        "Log": "log",
    }
    for source, target in replacements.items():
        code = code.replace(source, target)

    code = re.sub(r"\bsum\(", "ts_sum(", code)
    code = re.sub(r"\bstddev\(", "ts_stddev(", code)
    code = re.sub(r"\bcorrelation\(", "ts_corr(", code)
    code = re.sub(r"\bcovariance\(", "ts_cov(", code)
    code = re.sub(r"\bproduct\(", "ts_product(", code)
    code = code.replace("^", "**")
    code = code.replace("||", "|")
    code = code.replace("&&", "&")
    return code


class _FormulaValidator(ast.NodeVisitor):
    def __init__(self, alpha_id: int, constant_slots: dict[int, ConstantSlot]):
        self.alpha_id = int(alpha_id)
        self.constant_slots = constant_slots
        self.constant_index = 0

    def generic_visit(self, node):  # type: ignore[override]
        raise ValueError(f"Unsupported formula node: {type(node).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(
            node.op,
            ast.Add | ast.Sub | ast.Mult | ast.Div | ast.Pow | ast.BitAnd | ast.BitOr,
        ):
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, ast.UAdd | ast.USub | ast.Not | ast.Invert):
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, ast.And | ast.Or):
            raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")
        for value in node.values:
            self.visit(value)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            if not isinstance(op, ast.Lt | ast.LtE | ast.Gt | ast.GtE | ast.Eq | ast.NotEq):
                raise ValueError(f"Unsupported compare operator: {type(op).__name__}")
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed.")
        func_name = node.func.id
        if func_name not in CALL_NAMES:
            raise ValueError(f"Unsupported formula function: {func_name}")
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node: ast.Name) -> None:
        name = node.id
        if name in CALL_NAMES or name in ROOT_NAMES or _ADV_NAME_RE.match(name):
            return
        raise ValueError(f"Unsupported formula symbol: {name}")

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, int | float) or isinstance(node.value, bool):
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        value = float(node.value)
        if is_exempt_constant(value):
            return
        self.constant_index += 1
        key = f"alpha101.{self.alpha_id}.const.{self.constant_index:03d}"
        self.constant_slots[id(node)] = ConstantSlot(key=key, default=value)


def is_polars_supported(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int | float) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return node.id in ROOT_NAMES or bool(_ADV_NAME_RE.match(node.id))
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.UAdd | ast.USub):
            return False
        return is_polars_supported(node.operand)
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.Div | ast.Pow):
            return False
        return is_polars_supported(node.left) and is_polars_supported(node.right)
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return False
        if not isinstance(node.ops[0], ast.Lt | ast.LtE | ast.Gt | ast.GtE | ast.Eq | ast.NotEq):
            return False
        return is_polars_supported(node.left) and is_polars_supported(node.comparators[0])
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            return False
        name = node.func.id
        if name not in POLARS_CALLS:
            return False
        if name in {"max", "min"} and len(node.args) != 2:
            return False
        if name == "where" and len(node.args) != 3:
            return False
        if name in {"abs", "log", "sign"} and len(node.args) != 1:
            return False
        return all(is_polars_supported(arg) for arg in node.args)
    return False


def parse_formula_to_ir(alpha_id: int, expr: str) -> FormulaIR:
    normalized = normalize_formula(expr)
    parsed = ast.parse(normalized, mode="eval")
    if not isinstance(parsed, ast.Expression):
        raise ValueError("Formula parser expected expression tree.")
    slots: dict[int, ConstantSlot] = {}
    _FormulaValidator(alpha_id, slots).visit(parsed.body)
    return FormulaIR(
        alpha_id=int(alpha_id),
        source=str(expr),
        normalized=normalized,
        tree=parsed,
        constant_slots=slots,
    )
