from __future__ import annotations

import builtins

import lumina_quant.indicators.formulaic_alpha as formulaic_alpha
import pytest
from lumina_quant.indicators.formulaic_alpha import (
    clear_alpha101_param_overrides,
    compute_alpha101,
    list_alpha101_tunable_params,
    set_alpha101_param_overrides,
)


def _sample_payload(size: int = 256) -> dict[str, list[float]]:
    closes = [100.0 + (0.2 * i) + (0.05 if i % 2 == 0 else -0.03) for i in range(size)]
    opens = [value - 0.1 for value in closes]
    highs = [value + 0.3 for value in closes]
    lows = [value - 0.35 for value in closes]
    volumes = [1000.0 + (3.0 * i) for i in range(size)]
    vwaps = [
        ((high + low + close) / 3.0)
        for high, low, close in zip(highs, lows, closes, strict=False)
    ]
    return {
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "vwaps": vwaps,
    }


def test_formulaic_eval_path_does_not_use_builtin_eval(monkeypatch: pytest.MonkeyPatch):
    payload = _sample_payload()

    def _deny_eval(*_args, **_kwargs):
        raise AssertionError("eval() should not be called in formulaic runtime path")

    monkeypatch.setattr(builtins, "eval", _deny_eval)

    value = compute_alpha101(2, **payload)
    assert value is None or isinstance(float(value), float)


def test_alpha101_constant_registry_override_changes_output():
    payload = _sample_payload()
    params = list_alpha101_tunable_params(alpha_id=101)
    target_keys = sorted(params)
    assert target_keys
    target_key = target_keys[0]

    clear_alpha101_param_overrides()
    try:
        baseline = compute_alpha101(101, **payload)
        set_alpha101_param_overrides({target_key: 0.5})
        tuned = compute_alpha101(101, **payload)
    finally:
        clear_alpha101_param_overrides()

    assert baseline is not None
    assert tuned is not None
    assert tuned != baseline


def test_call_level_param_overrides_precede_global_registry():
    payload = _sample_payload()
    params = list_alpha101_tunable_params(alpha_id=101)
    target_key = sorted(params)[0]

    clear_alpha101_param_overrides()
    try:
        baseline = compute_alpha101(101, **payload)
        set_alpha101_param_overrides({target_key: 0.5})
        global_only = compute_alpha101(101, **payload)
        local_override = compute_alpha101(101, **payload, param_overrides={target_key: 0.001})
    finally:
        clear_alpha101_param_overrides()

    assert baseline is not None
    assert global_only is not None
    assert local_override is not None
    assert global_only != local_override
    assert abs(local_override - baseline) < abs(global_only - baseline)


def test_polars_backend_matches_numpy_for_simple_formula():
    if formulaic_alpha.pl is None:
        pytest.skip("polars not available")

    payload = _sample_payload()
    numpy_value = compute_alpha101(101, **payload, vector_backend="numpy")
    polars_value = compute_alpha101(101, **payload, vector_backend="polars")
    assert numpy_value is not None
    assert polars_value is not None
    assert pytest.approx(numpy_value, rel=1e-9, abs=1e-9) == polars_value


def test_tunable_params_are_keyed_by_alpha_namespace():
    params = list_alpha101_tunable_params(alpha_id=2)
    assert params
    assert all(key.startswith("alpha101.2.const.") for key in params)
