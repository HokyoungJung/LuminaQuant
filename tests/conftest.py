from __future__ import annotations

import importlib.util


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


_PUBLIC_ONLY_TESTS = [
    "test_accelerated_indicators.py",
    "test_advanced_alpha_indicators.py",
    "test_advanced_strategy_state.py",
    "test_alpha101_registry_compiler.py",
    "test_benchmark_pipeline_k_script.py",
    "test_compute_ops.py",
    "test_cost_aware_framework_e2e.py",
    "test_factory_candidate_set.py",
    "test_factory_fast_indicators.py",
    "test_formulaic_alpha_runtime.py",
    "test_formulaic_alpha_subset.py",
    "test_formulaic_definitions.py",
    "test_formulaic_definitions_specs.py",
    "test_formulaic_operators.py",
    "test_run_research_hurdle_script.py",
    "test_indicator_expansion.py",
    "test_indicators_core.py",
    "test_indicators_extended.py",
    "test_pair_trading_zscore.py",
    "test_rare_event_indicator.py",
    "test_rare_event_strategy.py",
    "test_scan_param_registry_script.py",
    "test_strategy_catalog_new_strategies.py",
    "test_strategy_long_short_support.py",
    "test_strategy_plugins_timeframe.py",
    "test_topcap_tsmom_strategy.py",
]

# Public repository publishes a reduced strategy/indicator surface.
# Ignore private-only tests when optional modules are not present.
if not (
    _has_module("lumina_quant.indicators.advanced_alpha")
    and _has_module("lumina_quant.indicators.formulaic_alpha")
    and _has_module("lumina_quant.strategies.moving_average")
    and _has_module("lumina_quant.strategies.factory_candidate_set")
):
    collect_ignore = list(_PUBLIC_ONLY_TESTS)
