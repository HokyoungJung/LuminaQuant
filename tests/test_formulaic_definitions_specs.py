from __future__ import annotations

from lumina_quant.indicators.formulaic_definitions import (
    ALPHA_FUNCTION_SPECS,
    get_alpha_function_spec,
    list_alpha_tunable_constants,
)
from lumina_quant.strategies.alpha101.compiler import build_context
from lumina_quant.strategies.alpha101.registry import (
    ALPHA_FUNCTION_SPECS as REGISTRY_ALPHA_FUNCTION_SPECS,
)


def _sample_context(size: int = 256):
    closes = [100.0 + (0.2 * idx) + (0.04 if idx % 2 == 0 else -0.03) for idx in range(size)]
    opens = [value - 0.1 for value in closes]
    highs = [value + 0.3 for value in closes]
    lows = [value - 0.35 for value in closes]
    volumes = [1000.0 + (3.0 * idx) for idx in range(size)]
    vwaps = [
        ((high + low + close) / 3.0)
        for high, low, close in zip(highs, lows, closes, strict=False)
    ]
    return build_context(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        vwaps=vwaps,
    )


def test_formulaic_definitions_exposes_specs_without_formula_string_table():
    expected_ids = list(range(1, 102))
    from lumina_quant.indicators import formulaic_definitions as defs

    assert not hasattr(defs, "ALPHA_FORMULAS")
    assert sorted(ALPHA_FUNCTION_SPECS) == expected_ids


def test_alpha_function_spec_exposes_callable_and_tunable_metadata():
    spec = get_alpha_function_spec(101)
    params = spec.tunable_constants
    assert params

    param_key = sorted(params)[0]
    assert param_key.startswith("alpha101.101.const.")

    context = _sample_context()
    baseline = spec.callable(context=context, vector_backend="numpy")
    tuned = spec.callable(
        context=context,
        param_overrides={param_key: 0.5},
        vector_backend="numpy",
    )

    assert baseline is not None
    assert tuned is not None
    assert tuned != baseline

    metadata = spec.metadata()
    assert metadata["alpha_id"] == 101
    assert metadata["constant_defaults"][param_key] == params[param_key]


def test_list_alpha_tunable_constants_reads_from_specs():
    spec = get_alpha_function_spec(2)
    params_from_spec = spec.tunable_constants
    params_from_listing = list_alpha_tunable_constants(alpha_id=2)

    assert params_from_listing == params_from_spec
    assert params_from_listing
    assert all(key.startswith("alpha101.2.const.") for key in params_from_listing)


def test_alpha101_registry_uses_formula_spec_objects():
    assert REGISTRY_ALPHA_FUNCTION_SPECS[101] is ALPHA_FUNCTION_SPECS[101]
