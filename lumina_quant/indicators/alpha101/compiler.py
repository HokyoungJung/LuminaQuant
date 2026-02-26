"""Alpha101 compiler + evaluator pipeline (parse -> IR -> compiled callable)."""

from __future__ import annotations

import ast
import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .formula_ir import FormulaIR, is_polars_supported, parse_formula_to_ir

try:  # Optional vectorized backend.
    import polars as pl
except Exception:  # pragma: no cover - optional dependency behavior
    pl = None


@dataclass(frozen=True, slots=True)
class CompiledAlphaFormula:
    ir: FormulaIR
    polars_capable: bool

    @property
    def alpha_id(self) -> int:
        return self.ir.alpha_id


def _last_finite_value(series: pd.Series) -> float | None:
    for value in reversed(series.to_list()):
        value_f = float(value)
        if math.isfinite(value_f):
            return value_f
    return None


@lru_cache(maxsize=1)
def _ops_module():
    return importlib.import_module("lumina_quant.indicators.formulaic_operators")


def _to_series(value, index: pd.Index) -> pd.Series:
    return _ops_module().to_series(value, index)


def build_context(
    *,
    opens,
    highs,
    lows,
    closes,
    volumes,
    vwaps=None,
    returns=None,
    cap=None,
    sector=None,
    industry=None,
    subindustry=None,
) -> dict[str, pd.Series]:
    n = max(
        len(closes),
        len(opens) if opens is not None else 0,
        len(highs) if highs is not None else 0,
        len(lows) if lows is not None else 0,
        len(volumes) if volumes is not None else 0,
        len(vwaps) if vwaps is not None else 0,
        len(returns) if returns is not None else 0,
    )
    index = pd.RangeIndex(start=0, stop=n, step=1)

    close_s = _to_series(closes, index)
    open_s = _to_series(opens if opens is not None else closes, index)
    high_s = _to_series(highs if highs is not None else closes, index)
    low_s = _to_series(lows if lows is not None else closes, index)
    volume_s = _to_series(volumes if volumes is not None else [0.0] * n, index)
    if vwaps is None:
        vwap_s = (high_s + low_s + close_s) / 3.0
    else:
        vwap_s = _to_series(vwaps, index)

    if returns is None:
        ret_s = close_s.pct_change().replace([np.inf, -np.inf], np.nan)
    else:
        ret_s = _to_series(returns, index)

    cap_s = _to_series(cap if cap is not None else [1.0] * n, index)
    sector_s = _to_series(sector if sector is not None else [0.0] * n, index)
    industry_s = _to_series(industry if industry is not None else [0.0] * n, index)
    subindustry_s = _to_series(subindustry if subindustry is not None else [0.0] * n, index)

    context: dict[str, pd.Series] = {
        "open": open_s,
        "high": high_s,
        "low": low_s,
        "close": close_s,
        "volume": volume_s,
        "vwap": vwap_s,
        "returns": ret_s,
        "cap": cap_s,
        "sector": sector_s,
        "industry": industry_s,
        "subindustry": subindustry_s,
    }

    for adv_window in (5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180):
        context[f"adv{adv_window}"] = (close_s * volume_s).rolling(adv_window).mean()

    return context


@lru_cache(maxsize=256)
def compile_formula(alpha_id: int, expr: str) -> CompiledAlphaFormula:
    ir = parse_formula_to_ir(alpha_id, expr)
    return CompiledAlphaFormula(ir=ir, polars_capable=is_polars_supported(ir.tree.body))


def _resolve_constant(
    node: ast.Constant,
    *,
    compiled: CompiledAlphaFormula,
    param_overrides: Mapping[str, float] | None,
    param_registry,
) -> float:
    base = float(node.value)
    slot = compiled.ir.constant_slots.get(id(node))
    if slot is None:
        return base
    if param_overrides is not None and slot.key in param_overrides:
        try:
            return float(param_overrides[slot.key])
        except (TypeError, ValueError):
            return slot.default
    if param_registry is None:
        return slot.default
    return float(param_registry.get(slot.key, slot.default))


def _bool_series(value, index: pd.Index) -> pd.Series:
    return _to_series(value, index).fillna(0.0).astype(bool)


def _apply_compare(op: ast.cmpop, left, right, *, index: pd.Index):
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")


def _eval_ast_node(
    node: ast.AST,
    *,
    env: dict[str, object],
    compiled: CompiledAlphaFormula,
    index: pd.Index,
    param_overrides: Mapping[str, float] | None,
    param_registry,
):
    if isinstance(node, ast.Constant):
        return _resolve_constant(
            node,
            compiled=compiled,
            param_overrides=param_overrides,
            param_registry=param_registry,
        )
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(
            node.operand,
            env=env,
            compiled=compiled,
            index=index,
            param_overrides=param_overrides,
            param_registry=param_registry,
        )
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.Not | ast.Invert):
            return ~_bool_series(operand, index)
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(
            node.left,
            env=env,
            compiled=compiled,
            index=index,
            param_overrides=param_overrides,
            param_registry=param_registry,
        )
        right = _eval_ast_node(
            node.right,
            env=env,
            compiled=compiled,
            index=index,
            param_overrides=param_overrides,
            param_registry=param_registry,
        )
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.BitAnd):
            return _bool_series(left, index) & _bool_series(right, index)
        if isinstance(node.op, ast.BitOr):
            return _bool_series(left, index) | _bool_series(right, index)
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
    if isinstance(node, ast.BoolOp):
        values = [
            _eval_ast_node(
                item,
                env=env,
                compiled=compiled,
                index=index,
                param_overrides=param_overrides,
                param_registry=param_registry,
            )
            for item in node.values
        ]
        if not values:
            return False
        result = _bool_series(values[0], index)
        for value in values[1:]:
            if isinstance(node.op, ast.And):
                result = result & _bool_series(value, index)
            elif isinstance(node.op, ast.Or):
                result = result | _bool_series(value, index)
            else:
                raise ValueError(f"Unsupported boolean op: {type(node.op).__name__}")
        return result
    if isinstance(node, ast.Compare):
        left = _eval_ast_node(
            node.left,
            env=env,
            compiled=compiled,
            index=index,
            param_overrides=param_overrides,
            param_registry=param_registry,
        )
        result = None
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            right = _eval_ast_node(
                comparator,
                env=env,
                compiled=compiled,
                index=index,
                param_overrides=param_overrides,
                param_registry=param_registry,
            )
            current = _apply_compare(op, left, right, index=index)
            result = (
                current
                if result is None
                else (_bool_series(result, index) & _bool_series(current, index))
            )
            left = right
        return result
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported callable node.")
        fn_name = node.func.id
        fn = env[fn_name]
        args = [
            _eval_ast_node(
                arg,
                env=env,
                compiled=compiled,
                index=index,
                param_overrides=param_overrides,
                param_registry=param_registry,
            )
            for arg in node.args
        ]
        kwargs = {
            kw.arg: _eval_ast_node(
                kw.value,
                env=env,
                compiled=compiled,
                index=index,
                param_overrides=param_overrides,
                param_registry=param_registry,
            )
            for kw in node.keywords
            if kw.arg is not None
        }
        return fn(*args, **kwargs)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _build_polars_expr(
    node: ast.AST,
    *,
    compiled: CompiledAlphaFormula,
    param_overrides: Mapping[str, float] | None,
    param_registry,
):
    if pl is None:
        raise RuntimeError("Polars backend unavailable.")
    if isinstance(node, ast.Constant):
        value = _resolve_constant(
            node,
            compiled=compiled,
            param_overrides=param_overrides,
            param_registry=param_registry,
        )
        return pl.lit(value)
    if isinstance(node, ast.Name):
        return pl.col(node.id)
    if isinstance(node, ast.UnaryOp):
        operand = _build_polars_expr(
            node.operand,
            compiled=compiled,
            param_overrides=param_overrides,
            param_registry=param_registry,
        )
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise NotImplementedError
    if isinstance(node, ast.BinOp):
        left = _build_polars_expr(
            node.left,
            compiled=compiled,
            param_overrides=param_overrides,
            param_registry=param_registry,
        )
        right = _build_polars_expr(
            node.right,
            compiled=compiled,
            param_overrides=param_overrides,
            param_registry=param_registry,
        )
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left.pow(right)
        raise NotImplementedError
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError
        left = _build_polars_expr(
            node.left,
            compiled=compiled,
            param_overrides=param_overrides,
            param_registry=param_registry,
        )
        right = _build_polars_expr(
            node.comparators[0],
            compiled=compiled,
            param_overrides=param_overrides,
            param_registry=param_registry,
        )
        op = node.ops[0]
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        raise NotImplementedError
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        args = [
            _build_polars_expr(
                arg,
                compiled=compiled,
                param_overrides=param_overrides,
                param_registry=param_registry,
            )
            for arg in node.args
        ]
        if name == "abs":
            return args[0].abs()
        if name == "log":
            return args[0].log()
        if name == "sign":
            return args[0].sign()
        if name == "where":
            return pl.when(args[0]).then(args[1]).otherwise(args[2])
        if name == "max":
            return pl.max_horizontal(args[0], args[1])
        if name == "min":
            return pl.min_horizontal(args[0], args[1])
    raise NotImplementedError


def evaluate_compiled_formula(
    compiled: CompiledAlphaFormula,
    context: dict[str, pd.Series],
    *,
    rank_window: int = 20,
    param_overrides: Mapping[str, float] | None = None,
    param_registry=None,
    vector_backend: str = "auto",
) -> float | None:
    index = next(iter(context.values())).index

    backend = str(vector_backend).strip().lower()
    if backend not in {"auto", "numpy", "polars"}:
        raise ValueError("vector_backend must be one of: auto, numpy, polars")

    if pl is not None and backend in {"auto", "polars"} and compiled.polars_capable:
        try:
            frame = pl.DataFrame({name: np.asarray(series, dtype=float) for name, series in context.items()})
            expr_pl = _build_polars_expr(
                compiled.ir.tree.body,
                compiled=compiled,
                param_overrides=param_overrides,
                param_registry=param_registry,
            )
            out = frame.lazy().select(expr_pl.alias("__alpha__")).collect()
            values = out["__alpha__"].to_numpy()
            result_series = pd.Series(values, index=index, dtype=float)
            result_series = result_series.replace([np.inf, -np.inf], np.nan)
            latest = _last_finite_value(result_series.dropna())
            if latest is not None:
                return latest
        except Exception:
            pass

    ops = _ops_module()
    env: dict[str, object] = {
        **context,
        "abs": np.abs,
        "log": np.log,
        "sign": np.sign,
        "rank": lambda s: ops.rank_series(s, index=index, window=max(2, int(rank_window))),
        "ts_rank": lambda s, w: ops.ts_rank_series(s, w, index=index),
        "ts_sum": lambda s, w: ops.ts_sum_series(s, w, index=index),
        "ts_stddev": lambda s, w: ops.ts_stddev_series(s, w, index=index),
        "ts_corr": lambda left, right, window: ops.ts_corr_series(left, right, window, index=index),
        "ts_cov": lambda left, right, window: ops.ts_cov_series(left, right, window, index=index),
        "ts_min": lambda s, w: ops.ts_min_series(s, w, index=index),
        "ts_max": lambda s, w: ops.ts_max_series(s, w, index=index),
        "ts_product": lambda s, w: ops.ts_product_series(s, w, index=index),
        "delay": lambda s, p=1: ops.delay_series(s, p, index=index),
        "delta": lambda s, p=1: ops.delta_series(s, p, index=index),
        "ts_argmax": lambda s, w: ops.ts_argmax_series(s, w, index=index),
        "ts_argmin": lambda s, w: ops.ts_argmin_series(s, w, index=index),
        "decay_linear": lambda s, p=10: ops.decay_linear_series(s, p, index=index),
        "scale": lambda s, a=1.0: ops.scale_series(
            s,
            rank_window=int(rank_window),
            index=index,
            a=a,
        ),
        "signed_power": lambda s, p: ops.signed_power_series(s, p, index=index),
        "where": lambda cond, left, right: ops.where_series(cond, left, right, index=index),
        "indneutralize": lambda s, g: ops.indneutralize_series(s, g, index=index),
        "max": np.maximum,
        "min": np.minimum,
    }
    result = _eval_ast_node(
        compiled.ir.tree.body,
        env=env,
        compiled=compiled,
        index=index,
        param_overrides=param_overrides,
        param_registry=param_registry,
    )
    result_series = _to_series(result, index).replace([np.inf, -np.inf], np.nan)
    if result_series.empty:
        return None
    latest = _last_finite_value(result_series.dropna())
    if latest is None:
        return None
    latest_float = float(latest)
    return latest_float if math.isfinite(latest_float) else None


def evaluate_formula(
    alpha_id: int,
    expr: str,
    context: dict[str, pd.Series],
    *,
    rank_window: int = 20,
    param_overrides: Mapping[str, float] | None = None,
    param_registry=None,
    vector_backend: str = "auto",
) -> float | None:
    compiled = compile_formula(alpha_id, expr)
    return evaluate_compiled_formula(
        compiled,
        context,
        rank_window=rank_window,
        param_overrides=param_overrides,
        param_registry=param_registry,
        vector_backend=vector_backend,
    )
