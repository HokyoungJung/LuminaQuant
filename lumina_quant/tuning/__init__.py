"""Hyper-parameter registry primitives."""

from .param_registry import (
    HyperParam,
    ParamRegistry,
    canonical_param_name,
    resolve_params_from_schema,
    strategy_slug,
)

__all__ = [
    "HyperParam",
    "ParamRegistry",
    "canonical_param_name",
    "resolve_params_from_schema",
    "strategy_slug",
]

