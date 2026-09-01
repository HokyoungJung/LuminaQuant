"""Safety gate: the live operator banner must never claim PAPER/TESTNET while
orders route to the PRODUCTION exchange endpoint.

Stage drives the endpoint (validate.py prod-routing gate: canary/full = prod,
require mode='real'). A banner that says PAPER while about to trade real money is
a dangerous operator-facing lie.
"""

from __future__ import annotations

from lumina_quant.cli.live import _resolve_live_banner
from lumina_quant.configuration.schema import LiveRuntimeConfig


def _banner(**kwargs):
    return _resolve_live_banner(LiveRuntimeConfig(**kwargs))


def test_testnet_stage_is_not_real_money():
    b = _banner(go_live_stage="testnet", mode="paper")
    assert b["endpoint_is_prod"] is False
    assert b["endpoint"] == "TESTNET"
    assert b["mode"] == "PAPER"


def test_shadow_stage_is_not_real_money():
    # Shadow is testnet-routed, no money at risk.
    b = _banner(go_live_stage="shadow", mode="real")
    assert b["endpoint_is_prod"] is False
    assert b["endpoint"] == "TESTNET"


def test_canary_stage_is_real_money():
    b = _banner(go_live_stage="canary", mode="real")
    assert b["endpoint_is_prod"] is True
    assert "PRODUCTION" in str(b["endpoint"])
    assert b["mode"] == "REAL"


def test_full_stage_is_real_money():
    b = _banner(go_live_stage="full", mode="real")
    assert b["endpoint_is_prod"] is True
    assert "PRODUCTION" in str(b["endpoint"])


def test_explicit_prod_endpoint_with_real_mode_is_real_money():
    # Even at an early stage, testnet=False + mode=real is a prod endpoint.
    b = _banner(go_live_stage="testnet", mode="real", testnet=False)
    assert b["endpoint_is_prod"] is True
    assert "PRODUCTION" in str(b["endpoint"])


def test_real_prod_config_never_displays_paper_or_testnet():
    """The exact lie the fix prevents: canary/full + real must not read PAPER/TESTNET."""
    for stage in ("canary", "full"):
        b = _banner(go_live_stage=stage, mode="real")
        assert b["mode"] != "PAPER"
        assert b["endpoint"] != "TESTNET"
        assert b["endpoint_is_prod"] is True
