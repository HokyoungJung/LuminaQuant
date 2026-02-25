from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parent.parent
    module_path = root / "scripts" / "scan_alpha101_params.py"
    spec = importlib.util.spec_from_file_location("scan_alpha101_params_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scan_alpha101_params module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_table_output_renders_headers_and_values():
    text = MODULE._table_output(
        {"alpha101.101.const.001": 0.001},
        {"alpha101.101.const.001": 0.25},
    )
    assert "| key | default | override |" in text
    assert "alpha101.101.const.001" in text
    assert "0.25" in text


def test_main_json_output_contains_param_namespace(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["scan_alpha101_params.py", "--alpha", "101"])
    MODULE.main()
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["alpha"] == 101
    assert payload["count"] >= 1
    assert any(key.startswith("alpha101.101.const.") for key in payload["params"])
