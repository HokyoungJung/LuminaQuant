"""Regression tests for sqrt_impact wiring and penalty clamp.

Covers:
- Penalty clamp prevents negative SELL fill price at extreme coefficient.
- Penalty clamp prevents BUY fill price > 2x raw even at extreme coefficient.
- sqrt_impact is now active through SimulatedExecutionHandler MKT path:
  a large MKT order pays more than a small one (not dead code anymore).
- flat model results are byte-identical when clamp is not triggered
  (normal penalties are well below 0.99 ceiling).
- Zero participation (bar_volume=0, adv=0) does not raise ZeroDivisionError.
"""

from __future__ import annotations

import math
import random
from typing import Any
from unittest.mock import MagicMock

import pytest

from lumina_quant.backtesting.execution_model import (
    ExecutionModel,
    ExecutionModelConfig,
)
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler


# ── helpers ──────────────────────────────────────────────────────────────────


def _flat_cfg(**overrides) -> ExecutionModelConfig:
    defaults = dict(
        taker_fee_rate=0.0004,
        maker_fee_rate=0.0002,
        slippage_rate=0.0010,
        spread_rate=0.0004,
        leverage=1,
        margin_mode="isolated",
        maintenance_margin_rate=0.005,
        liquidation_buffer_rate=0.0,
        funding_rate_per_8h=0.0,
        funding_interval_hours=8,
        random_seed=42,
        max_bar_volume_ratio=1.0,
        slippage_impact_model="flat",
        slippage_impact_coefficient=0.0,
        slippage_adv_quote=0.0,
    )
    defaults.update(overrides)
    return ExecutionModelConfig(**defaults)


def _sqrt_cfg(**overrides) -> ExecutionModelConfig:
    base = dict(
        taker_fee_rate=0.0004,
        maker_fee_rate=0.0002,
        slippage_rate=0.0010,
        spread_rate=0.0004,
        leverage=1,
        margin_mode="isolated",
        maintenance_margin_rate=0.005,
        liquidation_buffer_rate=0.0,
        funding_rate_per_8h=0.0,
        funding_interval_hours=8,
        random_seed=42,
        max_bar_volume_ratio=1.0,
        slippage_impact_model="sqrt_impact",
        slippage_impact_coefficient=0.01,
        slippage_adv_quote=0.0,
    )
    base.update(overrides)
    return ExecutionModelConfig(**base)


# ── clamp tests ───────────────────────────────────────────────────────────────


def test_clamp_prevents_negative_sell_fill_price():
    """A huge sqrt_impact coefficient must never produce fill_price <= 0 for SELL."""
    cfg = _sqrt_cfg(
        slippage_impact_coefficient=1000.0,  # absurd: would produce fill < 0 without clamp
        slippage_adv_quote=1.0,  # tiny ADV → huge participation
        random_seed=42,
    )
    em = ExecutionModel(cfg)
    raw = 50_000.0
    result = em.compute_fill(
        raw_price=raw,
        qty=1.0,
        direction="SELL",
        bar_volume=100.0,
        apply_liquidity_cap=False,
        order_notional=50_000.0,
    )
    assert result.fill_price > 0.0, f"fill_price={result.fill_price} must be positive; clamp failed"
    # Penalty capped at 0.99 → fill_price >= raw * (1 - 0.99) = raw * 0.01
    assert result.fill_price >= raw * 0.01 - 1e-6


def test_clamp_prevents_absurd_buy_fill_price():
    """A huge sqrt_impact coefficient must not produce fill_price > 2x raw for BUY."""
    cfg = _sqrt_cfg(
        slippage_impact_coefficient=1000.0,
        slippage_adv_quote=1.0,
        random_seed=42,
    )
    em = ExecutionModel(cfg)
    raw = 50_000.0
    result = em.compute_fill(
        raw_price=raw,
        qty=1.0,
        direction="BUY",
        bar_volume=100.0,
        apply_liquidity_cap=False,
        order_notional=50_000.0,
    )
    # Penalty capped at 0.99 → fill_price <= raw * 1.99
    assert result.fill_price <= raw * 1.99 + 1e-6


def test_clamp_commission_never_negative():
    """Commission must be >= 0 even with extreme penalty."""
    cfg = _sqrt_cfg(
        slippage_impact_coefficient=1000.0,
        slippage_adv_quote=1.0,
        random_seed=42,
    )
    em = ExecutionModel(cfg)
    result = em.compute_fill(
        raw_price=100.0,
        qty=1.0,
        direction="SELL",
        bar_volume=10.0,
        apply_liquidity_cap=False,
        order_notional=100.0,
    )
    assert result.commission >= 0.0, f"commission={result.commission} must be >= 0"


# ── flat model byte-identical (clamp not triggered) ────────────────────────────


def test_flat_model_byte_identical_after_clamp_added():
    """flat model normal-penalty fill is unchanged — clamp ceiling never reached."""
    seed = 77
    raw = 30_000.0
    bar_vol = 50.0

    cfg = _flat_cfg(random_seed=seed)
    em = ExecutionModel(cfg)
    result = em.compute_fill(
        raw_price=raw,
        qty=0.5,
        direction="BUY",
        bar_volume=bar_vol,
        apply_liquidity_cap=False,
    )

    # Recompute manually — same RNG, same arithmetic, clamp should be a no-op.
    rng = random.Random(seed)
    slip = rng.uniform(cfg.slippage_rate * 0.5, cfg.slippage_rate * 1.5)
    penalty = slip + cfg.spread_rate / 2.0
    # Verify penalty is well below ceiling before asserting price equality.
    assert penalty < 0.99, f"test precondition failed: normal penalty {penalty} exceeded ceiling"
    expected = raw * (1.0 + penalty)

    assert result.fill_price == pytest.approx(expected, rel=1e-12)


# ── zero-denominator safety ────────────────────────────────────────────────────


def test_sqrt_impact_zero_bar_volume_and_zero_adv_no_error():
    """bar_volume=0 and adv=0 → denominator=0 → impact=0, no ZeroDivisionError."""
    cfg = _sqrt_cfg(
        slippage_adv_quote=0.0,
        slippage_impact_coefficient=0.05,
        random_seed=42,
    )
    flat_cfg = _flat_cfg(random_seed=42)

    em_sqrt = ExecutionModel(cfg)
    em_flat = ExecutionModel(flat_cfg)

    result_sqrt = em_sqrt.compute_fill(
        raw_price=1000.0,
        qty=1.0,
        direction="BUY",
        bar_volume=0.0,  # zero volume → denominator=0 → impact path skipped
        apply_liquidity_cap=False,
        order_notional=1000.0,
    )
    result_flat = em_flat.compute_fill(
        raw_price=1000.0,
        qty=1.0,
        direction="BUY",
        bar_volume=0.0,
        apply_liquidity_cap=False,
        order_notional=1000.0,
    )
    # With zero denominator, sqrt_impact falls back to flat behaviour.
    assert result_sqrt.fill_price == pytest.approx(result_flat.fill_price, rel=1e-12)


# ── sqrt_impact wired via SimulatedExecutionHandler ───────────────────────────


def _make_sim_handler(cfg: ExecutionModelConfig, bar_open: float, bar_volume: float):
    """Build a SimulatedExecutionHandler backed by cfg, with a mock bars feed."""
    mock_bars = MagicMock()
    mock_bars.get_latest_bar_value.side_effect = lambda sym, field: {
        "open": bar_open,
        "high": bar_open * 1.01,
        "low": bar_open * 0.99,
        "close": bar_open,
        "volume": bar_volume,
    }[field]

    mock_events = MagicMock()
    mock_events.put = MagicMock()

    # Build a plain mock config that execution_sim accepts via _config_from_attrs.
    mock_cfg = MagicMock()
    mock_cfg._rt = None  # force _config_from_attrs path
    mock_cfg.RANDOM_SEED = cfg.random_seed
    mock_cfg.TAKER_FEE_RATE = cfg.taker_fee_rate
    mock_cfg.MAKER_FEE_RATE = cfg.maker_fee_rate
    mock_cfg.SLIPPAGE_RATE = cfg.slippage_rate
    mock_cfg.SPREAD_RATE = cfg.spread_rate
    mock_cfg.LEVERAGE = cfg.leverage
    mock_cfg.MARGIN_MODE = cfg.margin_mode
    mock_cfg.MAINTENANCE_MARGIN_RATE = cfg.maintenance_margin_rate
    mock_cfg.LIQUIDATION_BUFFER_RATE = cfg.liquidation_buffer_rate
    mock_cfg.FUNDING_RATE_PER_8H = cfg.funding_rate_per_8h
    mock_cfg.FUNDING_INTERVAL_HOURS = cfg.funding_interval_hours
    mock_cfg.SIM_MAX_BAR_VOLUME_RATIO = cfg.max_bar_volume_ratio
    mock_cfg.SLIPPAGE_IMPACT_MODEL = cfg.slippage_impact_model
    mock_cfg.SLIPPAGE_IMPACT_COEFFICIENT = cfg.slippage_impact_coefficient
    mock_cfg.SLIPPAGE_ADV_QUOTE = cfg.slippage_adv_quote
    mock_cfg.SIM_LATENCY_MIN_BARS = 1
    mock_cfg.SIM_LATENCY_MAX_BARS = 1

    handler = SimulatedExecutionHandler(mock_events, mock_bars, mock_cfg)
    return handler, mock_events


def _make_market_event(symbol: str, bar_open: float, bar_vol: float):
    ev = MagicMock()
    ev.type = "MARKET"
    ev.symbol = symbol
    ev.open = bar_open
    ev.high = bar_open * 1.01
    ev.low = bar_open * 0.99
    ev.close = bar_open
    ev.volume = bar_vol
    ev.time = 0
    return ev


def _make_order_event(symbol: str, qty: float, direction: str = "BUY"):
    ev = MagicMock()
    ev.type = "ORDER"
    ev.order_type = "MKT"
    ev.symbol = symbol
    ev.quantity = qty
    ev.direction = direction
    ev.position_side = "LONG" if direction == "BUY" else "SHORT"
    ev.reduce_only = False
    ev.client_order_id = None
    ev.stop_loss = None
    ev.take_profit = None
    ev.trailing_percent = None
    return ev


def test_sqrt_impact_wired_mkt_large_pays_more_than_small():
    """sqrt_impact is no longer dead: a large MKT order pays more than a small one."""
    adv = 500_000.0
    bar_open = 50_000.0
    bar_vol = 10_000.0  # large enough that max_bar_volume_ratio=1.0 doesn't cap either order

    cfg = _sqrt_cfg(
        slippage_adv_quote=adv,
        slippage_impact_coefficient=0.05,  # strong signal
        random_seed=42,
        max_bar_volume_ratio=1.0,
    )

    # Small order
    handler_small, events_small = _make_sim_handler(cfg, bar_open, bar_vol)
    order_small = _make_order_event("BTCUSDT", qty=0.001)
    handler_small.execute_order(order_small)
    market_ev = _make_market_event("BTCUSDT", bar_open, bar_vol)
    handler_small.check_open_orders(market_ev)
    fill_small = events_small.put.call_args[0][0]
    price_small = fill_small.fill_cost / fill_small.quantity

    # Large order — same seed, fresh handler
    handler_large, events_large = _make_sim_handler(cfg, bar_open, bar_vol)
    order_large = _make_order_event("BTCUSDT", qty=5.0)
    handler_large.execute_order(order_large)
    handler_large.check_open_orders(market_ev)
    fill_large = events_large.put.call_args[0][0]
    price_large = fill_large.fill_cost / fill_large.quantity

    assert price_large > price_small, (
        f"Large MKT order fill price {price_large:.4f} should exceed "
        f"small MKT order fill price {price_small:.4f} under sqrt_impact"
    )


def test_flat_model_mkt_unaffected_by_order_notional_wiring():
    """Wiring order_notional must not change flat model fill prices."""
    bar_open = 50_000.0
    bar_vol = 1000.0
    qty = 0.1

    cfg_flat = _flat_cfg(random_seed=42, max_bar_volume_ratio=1.0)

    # Direct compute_fill without order_notional (old caller style).
    em_no_notional = ExecutionModel(cfg_flat)
    res_no = em_no_notional.compute_fill(
        raw_price=bar_open,
        qty=qty,
        direction="BUY",
        bar_volume=bar_vol,
        apply_liquidity_cap=False,
        order_notional=None,
    )

    # Direct compute_fill with order_notional (new wired style).
    em_with_notional = ExecutionModel(cfg_flat)
    res_with = em_with_notional.compute_fill(
        raw_price=bar_open,
        qty=qty,
        direction="BUY",
        bar_volume=bar_vol,
        apply_liquidity_cap=False,
        order_notional=qty * bar_open,
    )

    assert res_no.fill_price == pytest.approx(res_with.fill_price, rel=1e-12), (
        "flat model: order_notional wiring must not change fill price"
    )
