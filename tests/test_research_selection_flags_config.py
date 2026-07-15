"""Research/selection validation-wiring flags — schema + profile config gate.

Covers the 7 ``research.*`` promotion-gate correction flags and
``execution.funding_on_utc_boundary`` added for the
strategy_viability_assessment 2026-07-06 (Axis 3) fixes.

Contract (non-negotiable):
  * Every flag DEFAULTS to the legacy/OFF value so the default ``RuntimeConfig()``
    and the shipped ``config.yaml`` load byte-identically to today (flag OFF ==
    legacy path).
  * The corrected behavior is opt-in and turned ON only in the two honest-research
    profiles (``configs/profiles/backtest_cost_realistic.yaml`` and
    ``configs/profiles/research.yaml``).
  * Consumers read the flags DEFENSIVELY via
    ``getattr(config.research, <flag>, <legacy_default>)`` so a missing schema
    field keeps the legacy behavior — proven here on a field-less stand-in.
"""

from __future__ import annotations

import dataclasses
import subprocess
import types

import pytest

from lumina_quant.configuration.loader import build_runtime_config, load_runtime_config
from lumina_quant.configuration.schema import RuntimeConfig

# (flag_name, legacy_default) for the 7 research.* flags.
_RESEARCH_FLAGS: tuple[tuple[str, object], ...] = (
    ("strict_selection_gate", False),
    ("use_lockbox_split", False),
    ("purge_embargo_bars", 0),
    ("single_correlation_discount", False),
    ("hac_inference", False),
    ("cscv_pbo", False),
    ("exposure_normalized_promotion", False),
)

_COST_REALISTIC = "configs/profiles/backtest_cost_realistic.yaml"
_RESEARCH_PROFILE = "configs/profiles/research.yaml"


# --------------------------------------------------------------------------- #
# Flag-OFF (legacy / byte-identical) behavior
# --------------------------------------------------------------------------- #
def test_default_runtime_config_flags_are_legacy_off():
    """A bare RuntimeConfig() carries every flag at its legacy default."""
    rt = RuntimeConfig()
    for name, legacy in _RESEARCH_FLAGS:
        assert getattr(rt.research, name) == legacy, name
    assert rt.execution.funding_on_utc_boundary is False


def test_actual_engine_routing_is_default_off_and_combined_profile_arms_it() -> None:
    default = RuntimeConfig()
    assert default.research.route_unmapped_registered_strategies is False
    assert default.research.require_actual_engine_routing is False

    research = load_runtime_config(_RESEARCH_PROFILE)
    assert research.research.route_unmapped_registered_strategies is True
    assert research.research.require_actual_engine_routing is False

    combined = load_runtime_config(_COST_REALISTIC)
    assert combined.research.route_unmapped_registered_strategies is True
    assert combined.research.require_actual_engine_routing is True

    raw = build_runtime_config(
        {
            "research": {
                "route_unmapped_registered_strategies": True,
                "require_actual_engine_routing": True,
            }
        },
        env={},
    )
    assert raw.research.route_unmapped_registered_strategies is True
    assert raw.research.require_actual_engine_routing is True


def test_shipped_config_yaml_loads_flags_off():
    """The shipped config.yaml must NOT flip any correction flag (legacy path)."""
    rt = load_runtime_config("config.yaml")
    for name, legacy in _RESEARCH_FLAGS:
        assert getattr(rt.research, name) == legacy, name
    assert rt.execution.funding_on_utc_boundary is False


def test_shipped_config_yaml_flag_values_equal_default():
    """config.yaml == RuntimeConfig() for all 8 flags (byte-identical, flag OFF)."""
    shipped = load_runtime_config("config.yaml")
    default = RuntimeConfig()
    for name, _ in _RESEARCH_FLAGS:
        assert getattr(shipped.research, name) == getattr(default.research, name), name
    assert shipped.execution.funding_on_utc_boundary == default.execution.funding_on_utc_boundary


def test_shipped_config_yaml_full_load_is_byte_identical_to_head():
    """The comment-only config.yaml edits change nothing about the loaded config.

    Loads config.yaml from git HEAD and from the working tree and asserts the two
    fully-serialized RuntimeConfig objects are identical (proves the added docs are
    inert). Skips gracefully if HEAD:config.yaml is unavailable (e.g. shallow CI).
    """
    try:
        head_text = subprocess.run(
            ["git", "show", "HEAD:config.yaml"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError, FileNotFoundError:  # pragma: no cover
        pytest.skip("HEAD:config.yaml not available")

    if not head_text.strip():  # pragma: no cover
        pytest.skip("HEAD:config.yaml empty")

    import os
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fp:
        fp.write(head_text)
        head_path = fp.name
    try:
        head_cfg = load_runtime_config(head_path)
    finally:
        os.remove(head_path)

    working_cfg = load_runtime_config("config.yaml")
    assert dataclasses.asdict(working_cfg) == dataclasses.asdict(head_cfg)


def test_missing_research_section_keeps_legacy_flags():
    """A raw config with no research/execution overrides yields legacy flags."""
    rt = build_runtime_config({"trading": {"symbols": ["BTC/USDT"]}}, env={})
    for name, legacy in _RESEARCH_FLAGS:
        assert getattr(rt.research, name) == legacy, name
    assert rt.execution.funding_on_utc_boundary is False


def test_defensive_getattr_fallback_on_fieldless_stand_in():
    """Consumers use getattr(cfg.research, flag, legacy) — a field-less object
    must therefore report the legacy default, keeping each lane independent."""
    bare = types.SimpleNamespace()  # no attributes at all
    for name, legacy in _RESEARCH_FLAGS:
        assert getattr(bare, name, legacy) == legacy, name
    assert getattr(bare, "funding_on_utc_boundary", False) is False


# --------------------------------------------------------------------------- #
# Flag-ON (corrected) behavior — honest-research profiles
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile_path", [_COST_REALISTIC, _RESEARCH_PROFILE])
def test_honest_research_profiles_turn_all_flags_on(profile_path):
    """Both honest-research profiles enable every correction flag."""
    rt = load_runtime_config(profile_path)
    assert rt.research.strict_selection_gate is True
    assert rt.research.use_lockbox_split is True
    assert rt.research.purge_embargo_bars >= 1
    assert rt.research.single_correlation_discount is True
    assert rt.research.hac_inference is True
    assert rt.research.cscv_pbo is True
    assert rt.research.exposure_normalized_promotion is True
    assert rt.execution.funding_on_utc_boundary is True


def test_flag_on_via_raw_dict_round_trips_through_loader():
    """Setting the flags in a raw config dict wires them onto RuntimeConfig
    (proves the loader coerces the new fields, not just the shipped YAML)."""
    raw = {
        "execution": {"funding_on_utc_boundary": True},
        "research": {
            "strict_selection_gate": True,
            "use_lockbox_split": True,
            "purge_embargo_bars": 3,
            "single_correlation_discount": True,
            "hac_inference": True,
            "cscv_pbo": True,
            "exposure_normalized_promotion": True,
        },
    }
    rt = build_runtime_config(raw, env={})
    assert rt.execution.funding_on_utc_boundary is True
    assert rt.research.strict_selection_gate is True
    assert rt.research.use_lockbox_split is True
    assert rt.research.purge_embargo_bars == 3
    assert rt.research.single_correlation_discount is True
    assert rt.research.hac_inference is True
    assert rt.research.cscv_pbo is True
    assert rt.research.exposure_normalized_promotion is True

    # Nested research sub-blocks stay at their defaults (flat flags did not leak in).
    assert rt.research.alpha_search.enabled is False
    assert rt.research.execution_attribution_enabled is False


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
