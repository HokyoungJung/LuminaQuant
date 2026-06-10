"""Phase 5 kill-switch envelope — negative tests for validate_runtime_config.

kill_switch_enabled=False and any disabled risk envelope field must be
REJECTED by _validate_kill_switch_and_risk_envelope, which runs first in
validate_runtime_config so no stage or mode can bypass the rejection.
"""

from __future__ import annotations

import pytest
from lumina_quant.configuration.loader import build_runtime_config
from lumina_quant.configuration.validate import validate_runtime_config


def _base_raw() -> dict:
    return {
        "trading": {"symbols": ["BTC/USDT"], "timeframe": "1m", "timeframes": ["1s", "1m"]},
        "storage": {
            "materializer_required_timeframes": ["1s", "1m"],
            "materializer_base_timeframe": "1s",
        },
        "live": {
            "mode": "paper",
            "market_data_source": "committed",
            "order_state_source": "polling",
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


# ── kill_switch_enabled ─────────────────────────────────────────────────────

def test_kill_switch_false_is_rejected():
    raw = _base_raw()
    raw["live"]["kill_switch_enabled"] = False
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="kill_switch_enabled"):
        validate_runtime_config(runtime)


def test_kill_switch_true_is_accepted():
    raw = _base_raw()
    raw["live"]["kill_switch_enabled"] = True
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)  # must not raise


def test_kill_switch_default_is_accepted():
    # Default kill_switch_enabled=True — omitting the key is fine
    raw = _base_raw()
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)


# ── max_daily_loss_pct ──────────────────────────────────────────────────────

def test_daily_loss_zero_is_rejected():
    raw = _base_raw()
    raw["risk"] = {"max_daily_loss_pct": 0.0}
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="max_daily_loss_pct"):
        validate_runtime_config(runtime)


def test_daily_loss_above_half_is_rejected():
    raw = _base_raw()
    raw["risk"] = {"max_daily_loss_pct": 0.51}
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="max_daily_loss_pct"):
        validate_runtime_config(runtime)


def test_daily_loss_exactly_half_is_accepted():
    raw = _base_raw()
    raw["risk"] = {"max_daily_loss_pct": 0.5}
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)  # must not raise


def test_daily_loss_small_positive_is_accepted():
    raw = _base_raw()
    raw["risk"] = {"max_daily_loss_pct": 0.01}
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)


# ── max_position_size_pct ───────────────────────────────────────────────────

def test_position_size_zero_is_rejected():
    raw = _base_raw()
    raw["risk"] = {"max_position_size_pct": 0.0}
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="max_position_size_pct"):
        validate_runtime_config(runtime)


def test_position_size_above_one_is_rejected():
    raw = _base_raw()
    raw["risk"] = {"max_position_size_pct": 1.01}
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="max_position_size_pct"):
        validate_runtime_config(runtime)


def test_position_size_exactly_one_is_accepted():
    raw = _base_raw()
    raw["risk"] = {"max_position_size_pct": 1.0}
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)


# ── consecutive_loss_halt_count ─────────────────────────────────────────────

def test_consecutive_loss_halt_zero_is_rejected():
    raw = _base_raw()
    raw["risk"] = {"consecutive_loss_halt_count": 0}
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="consecutive_loss_halt_count"):
        validate_runtime_config(runtime)


def test_consecutive_loss_halt_negative_is_rejected():
    raw = _base_raw()
    raw["risk"] = {"consecutive_loss_halt_count": -1}
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="consecutive_loss_halt_count"):
        validate_runtime_config(runtime)


def test_consecutive_loss_halt_positive_is_accepted():
    raw = _base_raw()
    raw["risk"] = {"consecutive_loss_halt_count": 3}
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)


# ── freeze_new_entries_on_breach in canary/full ─────────────────────────────

def test_freeze_false_in_canary_stage_is_rejected():
    raw = _base_raw()
    raw["live"]["go_live_stage"] = "canary"
    raw["risk"] = {"freeze_new_entries_on_breach": False}
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="freeze_new_entries_on_breach"):
        validate_runtime_config(runtime)


def test_freeze_false_in_full_stage_is_rejected():
    raw = _base_raw()
    raw["live"]["go_live_stage"] = "full"
    raw["risk"] = {"freeze_new_entries_on_breach": False}
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="freeze_new_entries_on_breach"):
        validate_runtime_config(runtime)


def test_freeze_false_in_testnet_stage_is_accepted():
    # freeze_new_entries_on_breach=False is only blocked in canary/full
    raw = _base_raw()
    raw["live"]["go_live_stage"] = "testnet"
    raw["risk"] = {"freeze_new_entries_on_breach": False}
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)


def test_freeze_false_in_shadow_stage_is_accepted():
    raw = _base_raw()
    raw["live"]["go_live_stage"] = "shadow"
    raw["risk"] = {"freeze_new_entries_on_breach": False}
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)


# ── go_live_stage validation ────────────────────────────────────────────────

def test_invalid_go_live_stage_is_rejected():
    raw = _base_raw()
    raw["live"]["go_live_stage"] = "staging"
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="go_live_stage"):
        validate_runtime_config(runtime)


def test_all_valid_go_live_stages_are_accepted():
    for stage in ("testnet", "shadow", "canary", "full"):
        raw = _base_raw()
        raw["live"]["go_live_stage"] = stage
        runtime = build_runtime_config(raw, env={})
        validate_runtime_config(runtime)


# ── canary_position_fraction ────────────────────────────────────────────────

def test_canary_position_fraction_zero_is_rejected():
    raw = _base_raw()
    raw["live"]["canary_position_fraction"] = 0.0
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="canary_position_fraction"):
        validate_runtime_config(runtime)


def test_canary_position_fraction_above_one_is_rejected():
    raw = _base_raw()
    raw["live"]["canary_position_fraction"] = 1.1
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="canary_position_fraction"):
        validate_runtime_config(runtime)


# ── shadow_parity_min_ratio ─────────────────────────────────────────────────

def test_shadow_parity_ratio_zero_is_rejected():
    raw = _base_raw()
    raw["live"]["shadow_parity_min_ratio"] = 0.0
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="shadow_parity_min_ratio"):
        validate_runtime_config(runtime)


def test_shadow_parity_window_bars_zero_is_rejected():
    raw = _base_raw()
    raw["live"]["shadow_parity_window_bars"] = 0
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match="shadow_parity_window_bars"):
        validate_runtime_config(runtime)
