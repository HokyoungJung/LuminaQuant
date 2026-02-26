from __future__ import annotations

import lumina_quant.strategies.alpha101.registry as alpha101_registry_module
from lumina_quant.strategies.alpha101.compiler import build_context
from lumina_quant.strategies.alpha101.formula_ir import is_exempt_constant, parse_formula_to_ir
from lumina_quant.strategies.alpha101.registry import (
    build_optuna_search_space,
    get_all_alpha_callables,
    list_alpha_ids,
    missing_alpha_ids,
)


def _sample_payload(size: int = 280) -> dict[str, list[float]]:
    closes = [100.0 + (0.2 * idx) + (0.05 if idx % 2 == 0 else -0.04) for idx in range(size)]
    opens = [value - 0.1 for value in closes]
    highs = [value + 0.35 for value in closes]
    lows = [value - 0.4 for value in closes]
    volumes = [1000.0 + (2.5 * idx) for idx in range(size)]
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


def test_alpha101_registry_has_full_coverage_and_callable_map():
    alpha_ids = list_alpha_ids()
    assert alpha_ids == tuple(range(1, 102))
    assert missing_alpha_ids() == ()

    callables = get_all_alpha_callables()
    assert sorted(callables) == list(range(1, 102))


def test_formula_ir_parameterizes_only_non_exempt_constants():
    ir = parse_formula_to_ir(
        101,
        "((close - open) / ((high - low) + 0.5 + 64 + 1e-9 + 0.03 + (-1)))",
    )
    slots = sorted(ir.constant_slots.values(), key=lambda slot: slot.key)

    assert is_exempt_constant(0.5)
    assert is_exempt_constant(64.0)
    assert is_exempt_constant(1e-9)
    assert is_exempt_constant(-1.0)

    assert len(slots) == 1
    assert slots[0].key.startswith("alpha101.101.const.")
    assert abs(slots[0].default - 0.03) <= 1e-12


def test_alpha101_optuna_search_space_is_wired_through_param_registry():
    config = build_optuna_search_space(alpha_id=101, n_trials=9)
    assert config["n_trials"] == 9
    assert config["params"]

    first_key = sorted(config["params"])[0]
    assert first_key.startswith("alpha101.101.const.")

    spec = config["params"][first_key]
    assert spec["type"] == "float"
    assert spec["low"] < spec["high"]


def test_alpha101_registry_callables_execute_for_all_ids():
    payload = _sample_payload()
    context = build_context(**payload)
    callables = get_all_alpha_callables()

    non_null = 0
    for alpha_id in range(1, 102):
        value = callables[alpha_id](context=context)
        assert value is None or isinstance(float(value), float)
        if value is not None:
            non_null += 1

    assert non_null >= 80


def test_registry_prefers_spec_callable_api_when_available(monkeypatch):
    payload = _sample_payload()
    context = build_context(**payload)

    class _StubSpec:
        formula = "(close)"

        @property
        def callable(self):
            def _call(**_kwargs):
                return 123.456

            return _call

    monkeypatch.setitem(alpha101_registry_module.ALPHA_FUNCTION_SPECS, 101, _StubSpec())
    value = alpha101_registry_module.evaluate_alpha(101, context=context, vector_backend="numpy")
    assert value == 123.456


def test_registry_reads_tunable_constant_metadata_from_specs(monkeypatch):
    class _MetaSpec:
        formula = "(close)"
        tunable_constants = {"alpha101.1.const.777": 7.77}

    monkeypatch.setitem(alpha101_registry_module.ALPHA_FUNCTION_SPECS, 1, _MetaSpec())
    params = alpha101_registry_module.list_tunable_params(alpha_id=1)
    assert params == {"alpha101.1.const.777": 7.77}
