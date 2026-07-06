"""Real-money live-safety config gate — real_money_readiness audit 2026-07-06.

Covers the new fail-closed validate gate (findings C1/C2/C3/C4/C6/M6) and the new
schema/config surface (M1/M3/M4). The gate triggers ONLY for a real-money Binance
live path (mode=='real' OR go_live_stage in {canary, full}); paper/testnet/shadow
and non-Binance drivers are unaffected, and the shipped default config stays
byte-identical.
"""

from __future__ import annotations

import pytest

from lumina_quant.configuration.loader import build_runtime_config
from lumina_quant.configuration.schema import LiveRuntimeConfig, RiskConfig
from lumina_quant.configuration.validate import validate_runtime_config
from lumina_quant.research_universe import BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED


def _real_raw(*, stage: str = "full", mode: str = "real") -> dict:
    """A fully-armed real-money Binance config that PASSES the safety gate."""
    return {
        "trading": {"symbols": ["BTC/USDT"], "timeframe": "1m", "timeframes": ["1s", "1m"]},
        "storage": {
            "materializer_required_timeframes": ["1s", "1m"],
            "materializer_base_timeframe": "1s",
        },
        "risk": {
            "auto_flatten_on_breach": True,
            "flatten_escalate_to_market": True,
        },
        "live": {
            "mode": mode,
            "go_live_stage": stage,
            "market_data_source": "committed",
            "order_state_source": "polling",
            "real_mode_managed_protective_stops": True,
            "max_limit_price_band_pct": 0.02,
            "equity_reconciliation_interval_sec": 30.0,
            "market_data_silence_timeout_sec": 45.0,
            "bar_sanity_check": True,
            "stale_symbol_after_ms": 5_000,
            "exchange": {
                "driver": "binance_futures",
                "name": "binance",
                "market_type": "future",
                "position_mode": "HEDGE",
                "margin_mode": "isolated",
                "leverage": 2,
            },
        },
    }


def _paper_raw() -> dict:
    raw = _real_raw()
    raw["live"]["mode"] = "paper"
    raw["live"]["go_live_stage"] = "testnet"
    # A paper config with NONE of the safety knobs must still validate.
    raw.pop("risk")
    for key in (
        "real_mode_managed_protective_stops",
        "max_limit_price_band_pct",
        "equity_reconciliation_interval_sec",
        "market_data_silence_timeout_sec",
        "bar_sanity_check",
        "stale_symbol_after_ms",
    ):
        raw["live"].pop(key, None)
    return raw


# ── schema defaults are OFF (byte-identical default config) ──────────────────


def test_new_schema_fields_default_off():
    risk = RiskConfig()
    assert risk.max_orders_per_minute == 0
    assert risk.max_daily_notional_turnover_pct == 0.0
    assert risk.max_position_age_hours == 0.0
    assert risk.enforce_gross_exposure_in_hedge is False

    live = LiveRuntimeConfig()
    assert live.max_bbo_spread_bps_at_submit == 0.0
    assert live.max_estimated_one_way_slippage_bps == 0.0
    assert live.require_bbo_for_limit_orders is False
    assert live.max_limit_price_band_pct == 0.0
    assert live.equity_reconciliation_interval_sec == 0.0
    assert live.market_data_silence_timeout_sec == 0.0
    assert live.require_state_fingerprint is False
    assert live.real_mode_managed_protective_stops is False


# ── gate does NOT trigger for paper/testnet/shadow ──────────────────────────


def test_paper_config_without_safety_knobs_validates():
    runtime = build_runtime_config(_paper_raw(), env={})
    validate_runtime_config(runtime)  # must not raise


@pytest.mark.parametrize("stage", ["testnet", "shadow"])
def test_testnet_and_shadow_paper_do_not_trigger_gate(stage):
    raw = _paper_raw()
    raw["live"]["go_live_stage"] = stage
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)  # must not raise


# ── fully-armed real config PASSES ──────────────────────────────────────────


@pytest.mark.parametrize("stage", ["canary", "full"])
def test_fully_armed_real_config_passes(stage):
    runtime = build_runtime_config(_real_raw(stage=stage), env={})
    validate_runtime_config(runtime)  # must not raise


def test_real_mode_testnet_stage_still_triggers_gate():
    # mode=real with stage=testnet also triggers the gate (mode is enough).
    raw = _real_raw(stage="testnet", mode="real")
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)  # armed → passes


# ── C1: reachable FLATTEN tier + market escalation ──────────────────────────


def test_c1_freeze_only_kill_switch_rejected():
    raw = _real_raw()
    raw["risk"] = {"auto_flatten_on_breach": False, "flatten_escalate_to_market": False}
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="FLATTEN"):
        validate_runtime_config(runtime)


def test_c1_flatten_tier_without_market_escalation_rejected():
    raw = _real_raw()
    raw["risk"] = {"auto_flatten_on_breach": True, "flatten_escalate_to_market": False}
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="flatten_escalate_to_market"):
        validate_runtime_config(runtime)


def test_c1_hard_drawdown_tier_satisfies_flatten_requirement():
    raw = _real_raw()
    raw["risk"] = {
        "auto_flatten_on_breach": False,
        "hard_drawdown_flatten_pct": 0.05,
        "flatten_escalate_to_market": True,
    }
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)  # hard tier + escalation → passes


# ── C2: managed protective stops ────────────────────────────────────────────


def test_c2_managed_protective_stops_required():
    raw = _real_raw()
    raw["live"]["real_mode_managed_protective_stops"] = False
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="real_mode_managed_protective_stops"):
        validate_runtime_config(runtime)


# ── C3: fat-finger / price-band guard ───────────────────────────────────────


def test_c3_no_price_guard_rejected():
    raw = _real_raw()
    raw["live"]["max_limit_price_band_pct"] = 0.0
    raw["live"]["require_bbo_for_limit_orders"] = False
    raw["live"]["book_ticker_enabled"] = False
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match=r"fat-finger|max_limit_price_band_pct"):
        validate_runtime_config(runtime)


def test_c3_bbo_plus_book_ticker_satisfies_guard():
    raw = _real_raw()
    raw["live"]["max_limit_price_band_pct"] = 0.0
    raw["live"]["require_bbo_for_limit_orders"] = True
    raw["live"]["book_ticker_enabled"] = True
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)  # bbo path → passes


def test_c3_require_bbo_without_book_ticker_rejected():
    raw = _real_raw()
    raw["live"]["max_limit_price_band_pct"] = 0.0
    raw["live"]["require_bbo_for_limit_orders"] = True
    raw["live"]["book_ticker_enabled"] = False
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match=r"fat-finger|book_ticker"):
        validate_runtime_config(runtime)


# ── C4: equity reconciliation interval ──────────────────────────────────────


def test_c4_equity_reconciliation_interval_required():
    raw = _real_raw()
    raw["live"]["equity_reconciliation_interval_sec"] = 0.0
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="equity_reconciliation_interval_sec"):
        validate_runtime_config(runtime)


# ── C6: market-data-silence watchdog ────────────────────────────────────────


def test_c6_market_data_silence_timeout_required():
    raw = _real_raw()
    raw["live"]["market_data_silence_timeout_sec"] = 0.0
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="market_data_silence_timeout_sec"):
        validate_runtime_config(runtime)


# ── M6: explicit symbols (never inherit the research universe) ───────────────


def test_m6_inherited_research_universe_rejected_in_real_mode():
    raw = _real_raw()
    raw["trading"]["symbols"] = list(BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED)
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match=r"research universe|explicit"):
        validate_runtime_config(runtime)


def test_m6_explicit_small_symbol_list_accepted():
    raw = _real_raw()
    raw["trading"]["symbols"] = ["BTC/USDT", "ETH/USDT"]
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)  # explicit small list → passes


def test_m6_default_config_inherits_universe_and_is_rejected_in_real_mode():
    # Omitting trading.symbols inherits the full research universe default; a real
    # config must never do this.
    raw = _real_raw()
    raw["trading"].pop("symbols")
    runtime = build_runtime_config(raw, env={})
    assert set(runtime.trading.symbols) == set(BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED)
    with pytest.raises(ValueError, match=r"research universe|explicit"):
        validate_runtime_config(runtime)


# ── always-on non-negativity bounds for the new numeric fields ──────────────


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("risk", "max_orders_per_minute", -1),
        ("risk", "max_daily_notional_turnover_pct", -0.1),
        ("risk", "max_position_age_hours", -1.0),
        ("live", "max_bbo_spread_bps_at_submit", -1.0),
        ("live", "max_estimated_one_way_slippage_bps", -1.0),
        ("live", "max_limit_price_band_pct", -0.01),
        ("live", "equity_reconciliation_interval_sec", -1.0),
        ("live", "market_data_silence_timeout_sec", -1.0),
    ],
)
def test_new_fields_reject_negative_values_even_in_paper_mode(section, field, value):
    raw = _paper_raw()
    raw.setdefault(section, {})[field] = value
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match=field):
        validate_runtime_config(runtime)
