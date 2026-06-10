"""Integration tests for Phase 1 unified config entry-point.

Verifies that there is exactly one config entry point
(``lumina_quant.configuration``) and that the Phase 1 knobs
(``memory.cap_gb`` and ``validation.golden_rtol``) round-trip through YAML.
"""

from __future__ import annotations

import textwrap


def test_single_config_entry_point():
    """``get_default_runtime_config`` must come from ``lumina_quant.configuration``.

    No other public module should expose a competing top-level config loader.
    """
    import lumina_quant.configuration as cfg_mod

    assert hasattr(cfg_mod, "get_default_runtime_config")
    assert hasattr(cfg_mod, "load_runtime_config")
    assert hasattr(cfg_mod, "validate_runtime_config")

    # The old metaclass runtime_access module must not exist.
    import importlib
    import sys

    sys.modules.pop("lumina_quant.configuration.runtime_access", None)
    try:
        importlib.import_module("lumina_quant.configuration.runtime_access")
        raise AssertionError("runtime_access.py still present — must be deleted")
    except ModuleNotFoundError:
        pass  # expected


def test_memory_knob_default():
    """RuntimeConfig.memory.cap_gb defaults to 8.0 (Phase 1 knob)."""
    from lumina_quant.configuration.schema import MemoryConfig, RuntimeConfig

    rt = RuntimeConfig()
    assert hasattr(rt, "memory")
    assert isinstance(rt.memory, MemoryConfig)
    assert rt.memory.cap_gb == 8.0


def test_validation_knob_default():
    """RuntimeConfig.validation.golden_rtol defaults to 1e-8 (Phase 1 knob)."""
    from lumina_quant.configuration.schema import RuntimeConfig, ValidationConfig

    rt = RuntimeConfig()
    assert hasattr(rt, "validation")
    assert isinstance(rt.validation, ValidationConfig)
    assert rt.validation.golden_rtol == 1e-8


def test_memory_knob_yaml_round_trip(tmp_path, monkeypatch):
    """memory.cap_gb is loaded from YAML when present."""
    cfg = textwrap.dedent(
        """
        memory:
          cap_gb: 16.0
        """
    ).strip()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    monkeypatch.setenv("LQ_CONFIG_PATH", str(cfg_path))

    from lumina_quant.configuration import get_default_runtime_config

    rt = get_default_runtime_config()
    assert rt.memory.cap_gb == 16.0


def test_validation_knob_yaml_round_trip(tmp_path, monkeypatch):
    """validation.golden_rtol is loaded from YAML when present."""
    cfg = textwrap.dedent(
        """
        validation:
          golden_rtol: 1.0e-6
        """
    ).strip()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    monkeypatch.setenv("LQ_CONFIG_PATH", str(cfg_path))

    from lumina_quant.configuration import get_default_runtime_config

    rt = get_default_runtime_config()
    assert abs(rt.validation.golden_rtol - 1e-6) < 1e-15


def test_missing_config_file_returns_defaults(tmp_path, monkeypatch):
    """get_default_runtime_config() returns schema defaults when file is absent."""
    monkeypatch.setenv("LQ_CONFIG_PATH", str(tmp_path / "does_not_exist.yaml"))

    from lumina_quant.configuration import get_default_runtime_config
    from lumina_quant.configuration.schema import RuntimeConfig

    rt = get_default_runtime_config()
    assert isinstance(rt, RuntimeConfig)
    assert rt.trading.timeframe == "1m"
    assert rt.memory.cap_gb == 8.0
    assert rt.validation.golden_rtol == 1e-8
