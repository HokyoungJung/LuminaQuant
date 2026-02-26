"""Alpha101 formula IR/compiler/registry toolkit."""

from .compiler import (
    CompiledAlphaFormula,
    build_context,
    compile_formula,
    evaluate_compiled_formula,
)
from .formula_ir import (
    ConstantSlot,
    FormulaIR,
    is_exempt_constant,
    normalize_formula,
    parse_formula_to_ir,
)
from .registry import (
    ALPHA101_PARAM_REGISTRY,
    build_optuna_search_space,
    clear_param_overrides,
    evaluate_alpha,
    get_all_alpha_callables,
    get_alpha_callable,
    get_compiled_formula,
    get_formula,
    list_alpha_ids,
    list_tunable_params,
    missing_alpha_ids,
    set_param_overrides,
)

__all__ = [
    "ALPHA101_PARAM_REGISTRY",
    "CompiledAlphaFormula",
    "ConstantSlot",
    "FormulaIR",
    "build_context",
    "build_optuna_search_space",
    "clear_param_overrides",
    "compile_formula",
    "evaluate_alpha",
    "evaluate_compiled_formula",
    "get_all_alpha_callables",
    "get_alpha_callable",
    "get_compiled_formula",
    "get_formula",
    "is_exempt_constant",
    "list_alpha_ids",
    "list_tunable_params",
    "missing_alpha_ids",
    "normalize_formula",
    "parse_formula_to_ir",
    "set_param_overrides",
]
