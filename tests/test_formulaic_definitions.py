from __future__ import annotations

from lumina_quant.indicators import formulaic_definitions as defs


def test_formulaic_definitions_exposes_code_specs_only():
    assert not hasattr(defs, "ALPHA_FORMULAS")
    assert sorted(defs.ALPHA_FUNCTION_SPECS) == list(range(1, 102))


def test_alpha_function_specs_coverage_and_formula_alignment():
    assert sorted(defs.ALPHA_FUNCTION_SPECS) == list(range(1, 102))

    spec = defs.get_alpha_function_spec(58)
    assert spec.alpha_id == 58
    assert callable(spec.callable)
    assert "alpha101.58.const." in " ".join(spec.tunable_constants)


def test_alpha_spec_exposes_tunable_constant_metadata():
    spec = defs.get_alpha_function_spec(101)
    constants = spec.tunable_constants
    assert constants
    assert sorted(constants)[0].startswith("alpha101.101.const.")
    assert abs(constants["alpha101.101.const.001"] - 0.001) <= 1e-12

    via_helper = defs.list_alpha_tunable_constants(alpha_id=101)
    assert via_helper == constants
