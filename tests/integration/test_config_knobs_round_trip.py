"""Round-trip + discoverability gate for the spec's config-level knobs.

Spec constraint: "config-driven UX — everything the user touches is controlled
at the config level". This test gates two things the audit found missing:

1. Every spec knob loads from YAML through the single loader (round-trip).
2. The SHIPPED config.yaml and config.example.yaml actually expose the knobs,
   so a user reading them can discover the go-live stage, canary sizing, memory
   cap, golden tolerance, and data kinds.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (yaml dotted path within the named section file, ...) the knobs must be present in.
_SPEC_LIVE_KNOBS = (
    "go_live_stage",
    "kill_switch_enabled",
    "canary_position_fraction",
    "shadow_parity_min_ratio",
    "shadow_parity_window_bars",
)


def test_all_spec_knobs_round_trip_from_yaml(tmp_path, monkeypatch):
    """Non-default values for every spec knob bind through the loader."""
    cfg = textwrap.dedent(
        """
        live:
          go_live_stage: canary
          kill_switch_enabled: true
          canary_position_fraction: 0.25
          shadow_parity_min_ratio: 0.95
          shadow_parity_window_bars: 500
        memory:
          cap_gb: 12.0
        validation:
          golden_rtol: 1.0e-6
        data:
          kinds: [ohlcv]
          tick_path: /tmp/ticks
        """
    ).strip()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    monkeypatch.setenv("LQ_CONFIG_PATH", str(cfg_path))

    from lumina_quant.configuration import get_default_runtime_config

    rt = get_default_runtime_config()

    assert rt.live.go_live_stage == "canary"
    assert rt.live.kill_switch_enabled is True
    assert rt.live.canary_position_fraction == 0.25
    assert rt.live.shadow_parity_min_ratio == 0.95
    assert rt.live.shadow_parity_window_bars == 500
    assert rt.memory.cap_gb == 12.0
    assert abs(rt.validation.golden_rtol - 1e-6) < 1e-15
    assert rt.data.kinds == ["ohlcv"]
    assert rt.data.tick_path == "/tmp/ticks"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_shipped_root_config_exposes_spec_knobs():
    """Root config.yaml must surface every spec knob (discoverability)."""
    data = _load_yaml(PROJECT_ROOT / "config.yaml")
    live = data.get("live", {})
    for knob in _SPEC_LIVE_KNOBS:
        assert knob in live, f"config.yaml live.{knob} missing"
    assert "cap_gb" in data.get("memory", {}), "config.yaml memory.cap_gb missing"
    assert "golden_rtol" in data.get("validation", {}), "config.yaml validation.golden_rtol missing"
    assert "kinds" in data.get("data", {}), "config.yaml data.kinds missing"


def test_shipped_example_config_exposes_spec_knobs():
    """config.example.yaml must surface every spec knob (discoverability)."""
    data = _load_yaml(PROJECT_ROOT / "configs" / "config.example.yaml")
    live = data.get("live", {})
    for knob in _SPEC_LIVE_KNOBS:
        assert knob in live, f"config.example.yaml live.{knob} missing"
    assert "cap_gb" in data.get("memory", {}), "config.example.yaml memory.cap_gb missing"
    assert "golden_rtol" in data.get("validation", {}), (
        "config.example.yaml validation.golden_rtol missing"
    )
    assert "kinds" in data.get("data", {}), "config.example.yaml data.kinds missing"
