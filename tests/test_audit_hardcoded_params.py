from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parent.parent
    module_path = root / "scripts" / "audit_hardcoded_params.py"
    spec = importlib.util.spec_from_file_location("audit_hardcoded_params", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load audit_hardcoded_params module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_exemption_values_match_expected_contract():
    assert MODULE.is_exempt_value(0)
    assert MODULE.is_exempt_value(1)
    assert MODULE.is_exempt_value(-1)
    assert MODULE.is_exempt_value(0.5)
    assert MODULE.is_exempt_value(64)
    assert MODULE.is_exempt_value(1e-12)
    assert MODULE.is_exempt_value(1e-9)
    assert MODULE.is_exempt_value(1e-6)

    assert not MODULE.is_exempt_value(3)
    assert not MODULE.is_exempt_value(0.03)
    assert not MODULE.is_exempt_value(257)


def test_detects_non_exempt_numeric_literals(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)

    sample = tmp_path / "lumina_quant" / "strategies" / "sample.py"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(
        """

def f(threshold=3, decay=0.5):
    alpha = 257
    beta = -1
    eps = 1e-9
    return threshold + alpha + beta + eps
""",
        encoding="utf-8",
    )

    plan = MODULE._build_scan_plan(["lumina_quant/strategies"], [])
    violations = MODULE.collect_violations(plan)

    values = sorted({float(item.value) for item in violations})
    assert 3.0 in values
    assert 257.0 in values
    assert -1.0 not in values
    assert 0.5 not in values
    assert 1e-9 not in values


def test_detects_numeric_constants_inside_formula_strings(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)

    formula_file = tmp_path / "lumina_quant" / "indicators" / "formulaic_definitions.py"
    formula_file.parent.mkdir(parents=True, exist_ok=True)
    formula_file.write_text(
        """
ALPHA_FORMULAS = {
    1: "rank(delta(close, 7) + 0.03 - 1e-9 + 0.5)",
}
""",
        encoding="utf-8",
    )

    plan = MODULE._build_scan_plan([], ["lumina_quant/indicators/formulaic_definitions.py"])
    violations = MODULE.collect_violations(plan)

    assert violations
    assert {item.kind for item in violations} == {"formula_string"}
    values = sorted({float(item.value) for item in violations})
    assert 7.0 in values
    assert 0.03 in values
    assert 0.5 not in values
    assert 1e-9 not in values
    assert any("alpha_1" in item.suggestion for item in violations)


def test_skips_formula_string_literals_when_tunable_specs_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)

    formula_file = tmp_path / "lumina_quant" / "indicators" / "formulaic_definitions.py"
    formula_file.parent.mkdir(parents=True, exist_ok=True)
    formula_file.write_text(
        """
ALPHA_FORMULAS = {
    1: "rank(delta(close, 7) + 0.03 - 1e-9 + 0.5)",
}
ALPHA_FUNCTION_SPECS = {}
""",
        encoding="utf-8",
    )

    plan = MODULE._build_scan_plan([], ["lumina_quant/indicators/formulaic_definitions.py"])
    violations = MODULE.collect_violations(plan)

    assert violations == []


def test_baseline_mode_filters_existing_violations(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)

    sample = tmp_path / "lumina_quant" / "strategies" / "sample.py"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("x = 3\n", encoding="utf-8")

    args = [
        "--literal-paths",
        "lumina_quant/strategies",
        "--formula-paths",
        "lumina_quant/strategies",
        "--baseline",
        "reports/quality/baseline.json",
        "--json-out",
        "reports/quality/report.json",
        "--md-out",
        "reports/quality/report.md",
    ]

    # No baseline yet -> should fail on new violation.
    assert MODULE.main(args) == 1

    # Write baseline, then re-run should pass.
    assert MODULE.main([*args, "--write-baseline"]) == 0
    assert MODULE.main(args) == 0

    # If configured, baselined entries can be forced to fail.
    assert MODULE.main([*args, "--fail-on-baselined"]) == 1


def test_formula_signature_is_stable_across_line_shifts(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)

    formula_path = tmp_path / "lumina_quant" / "indicators" / "formulaic_definitions.py"
    formula_path.parent.mkdir(parents=True, exist_ok=True)

    base_formula = """
ALPHA_FORMULAS = {
    1: "rank(delta(close, 7) + 0.03)",
}
"""
    shifted_formula = """
# header shift
ALPHA_FORMULAS = {
    1: "rank(delta(close, 7) + 0.03)",
}
"""

    formula_path.write_text(base_formula, encoding="utf-8")
    base_violations = MODULE.collect_violations(
        MODULE._build_scan_plan([], ["lumina_quant/indicators/formulaic_definitions.py"])
    )
    base_signatures = {item.signature() for item in base_violations}

    formula_path.write_text(shifted_formula, encoding="utf-8")
    shifted_violations = MODULE.collect_violations(
        MODULE._build_scan_plan([], ["lumina_quant/indicators/formulaic_definitions.py"])
    )
    shifted_signatures = {item.signature() for item in shifted_violations}

    assert base_signatures
    assert base_signatures == shifted_signatures


def test_formula_signature_accepts_legacy_baseline_signatures(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)

    formula_path = tmp_path / "lumina_quant" / "indicators" / "formulaic_definitions.py"
    formula_path.parent.mkdir(parents=True, exist_ok=True)
    formula_path.write_text(
        """
ALPHA_FORMULAS = {
    1: "rank(delta(close, 7) + 0.03)",
}
""",
        encoding="utf-8",
    )

    plan = MODULE._build_scan_plan([], ["lumina_quant/indicators/formulaic_definitions.py"])
    violations = MODULE.collect_violations(plan)
    assert violations

    legacy_signatures = sorted(item.legacy_signature() for item in violations)
    baseline_path = tmp_path / "reports" / "quality" / "baseline.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps({"signatures": legacy_signatures}, indent=2) + "\n",
        encoding="utf-8",
    )

    rc = MODULE.main(
        [
            "--literal-paths",
            "lumina_quant/strategies",
            "--formula-paths",
            "lumina_quant/indicators/formulaic_definitions.py",
            "--baseline",
            "reports/quality/baseline.json",
            "--json-out",
            "reports/quality/report.json",
            "--md-out",
            "reports/quality/report.md",
        ]
    )
    assert rc == 0
