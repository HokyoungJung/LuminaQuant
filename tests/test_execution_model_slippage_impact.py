"""Audit finding fix: size/impact-aware slippage model (sqrt_impact).

Tests verify:
- "flat" model (default) reproduces the legacy fill price exactly — same RNG sequence.
- "sqrt_impact" model makes larger orders pay strictly more slippage than smaller ones.
- Impact is zero when slippage_impact_coefficient == 0 (regardless of model name).
- Impact is zero when order_notional is None (backward-compat — flat fallback).
- Impact uses slippage_adv_quote as denominator when > 0, else bar_volume * raw_price.
"""

from __future__ import annotations

import math
import random

import pytest

from lumina_quant.backtesting.execution_model import (
    ExecutionModel,
    ExecutionModelConfig,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _flat_cfg(**overrides) -> ExecutionModelConfig:
    """Return a default (flat) ExecutionModelConfig, optionally overriding fields."""
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
        max_bar_volume_ratio=1.0,  # allow full bar volume
        slippage_impact_model="flat",
        slippage_impact_coefficient=0.0,
        slippage_adv_quote=0.0,
    )
    defaults.update(overrides)
    return ExecutionModelConfig(**defaults)


def _sqrt_cfg(**overrides) -> ExecutionModelConfig:
    """Return a sqrt_impact ExecutionModelConfig."""
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


# ── "flat" model reproduces legacy fill price exactly ────────────────────────


def test_flat_model_buy_fill_matches_manual_rng():
    """flat model fill price == raw_price * (1 + slip + spread/2), same RNG draw."""
    seed = 7
    cfg = _flat_cfg(random_seed=seed)
    em = ExecutionModel(cfg)

    raw = 50_000.0
    bar_vol = 10.0

    result = em.compute_fill(
        raw_price=raw,
        qty=1.0,
        direction="BUY",
        bar_volume=bar_vol,
        apply_liquidity_cap=False,
    )

    # Recompute manually with the same seed
    rng = random.Random(seed)
    slip = rng.uniform(cfg.slippage_rate * 0.5, cfg.slippage_rate * 1.5)
    expected_penalty = slip + cfg.spread_rate / 2.0
    expected_price = raw * (1.0 + expected_penalty)

    assert result.fill_price == pytest.approx(expected_price, rel=1e-12)


def test_flat_model_sell_fill_matches_manual_rng():
    """flat model SELL fill price == raw_price * (1 - penalty)."""
    seed = 99
    cfg = _flat_cfg(random_seed=seed)
    em = ExecutionModel(cfg)

    raw = 3_000.0
    result = em.compute_fill(
        raw_price=raw,
        qty=0.5,
        direction="SELL",
        bar_volume=100.0,
        apply_liquidity_cap=False,
    )

    rng = random.Random(seed)
    slip = rng.uniform(cfg.slippage_rate * 0.5, cfg.slippage_rate * 1.5)
    expected_penalty = slip + cfg.spread_rate / 2.0
    expected_price = raw * (1.0 - expected_penalty)

    assert result.fill_price == pytest.approx(expected_price, rel=1e-12)


def test_flat_model_with_order_notional_ignores_notional():
    """flat model must produce the SAME fill price whether order_notional is passed or not."""
    cfg = _flat_cfg(random_seed=5)

    em_no_notional = ExecutionModel(cfg)
    res_no = em_no_notional.compute_fill(
        raw_price=1000.0,
        qty=1.0,
        direction="BUY",
        bar_volume=500.0,
        apply_liquidity_cap=False,
        order_notional=None,
    )

    em_with_notional = ExecutionModel(cfg)  # fresh instance, same seed
    res_with = em_with_notional.compute_fill(
        raw_price=1000.0,
        qty=1.0,
        direction="BUY",
        bar_volume=500.0,
        apply_liquidity_cap=False,
        order_notional=999_999.0,  # large notional, must be ignored on flat model
    )

    assert res_no.fill_price == res_with.fill_price


# ── sqrt_impact: larger order pays strictly more ──────────────────────────────


def test_sqrt_impact_larger_order_pays_more_with_adv_denominator():
    """With slippage_adv_quote set, a larger order pays strictly more than a smaller one."""
    adv = 1_000_000.0  # $1M ADV
    cfg = _sqrt_cfg(slippage_adv_quote=adv, slippage_impact_coefficient=0.01, random_seed=42)

    raw = 50_000.0

    # Both instances share the same seed so the flat component is identical.
    em_small = ExecutionModel(cfg)
    res_small = em_small.compute_fill(
        raw_price=raw,
        qty=0.01,
        direction="BUY",
        bar_volume=100.0,
        apply_liquidity_cap=False,
        order_notional=500.0,  # small order: 500 / 1e6 = 0.05% participation
    )

    em_large = ExecutionModel(cfg)
    res_large = em_large.compute_fill(
        raw_price=raw,
        qty=1.0,
        direction="BUY",
        bar_volume=100.0,
        apply_liquidity_cap=False,
        order_notional=50_000.0,  # large order: 50_000 / 1e6 = 5% participation
    )

    assert res_large.fill_price > res_small.fill_price, (
        f"Large order fill {res_large.fill_price} should exceed small order fill {res_small.fill_price}"
    )


def test_sqrt_impact_larger_order_pays_more_with_bar_volume_denominator():
    """Without adv, denominator = bar_volume * raw_price; larger notional → more impact."""
    cfg = _sqrt_cfg(slippage_adv_quote=0.0, slippage_impact_coefficient=0.01, random_seed=42)

    raw = 1_000.0
    bar_vol = 100.0  # bar quote volume = 100 * 1000 = 100_000

    em_small = ExecutionModel(cfg)
    res_small = em_small.compute_fill(
        raw_price=raw,
        qty=1.0,
        direction="BUY",
        bar_volume=bar_vol,
        apply_liquidity_cap=False,
        order_notional=100.0,  # participation = 100 / 100_000 = 0.1%
    )

    em_large = ExecutionModel(cfg)
    res_large = em_large.compute_fill(
        raw_price=raw,
        qty=10.0,
        direction="BUY",
        bar_volume=bar_vol,
        apply_liquidity_cap=False,
        order_notional=10_000.0,  # participation = 10_000 / 100_000 = 10%
    )

    assert res_large.fill_price > res_small.fill_price


def test_sqrt_impact_sell_larger_order_pays_more():
    """SELL direction: larger order fill price is strictly LOWER (more adversarial)."""
    adv = 500_000.0
    cfg = _sqrt_cfg(slippage_adv_quote=adv, slippage_impact_coefficient=0.01, random_seed=42)

    raw = 2_000.0

    em_small = ExecutionModel(cfg)
    res_small = em_small.compute_fill(
        raw_price=raw,
        qty=0.1,
        direction="SELL",
        bar_volume=50.0,
        apply_liquidity_cap=False,
        order_notional=200.0,
    )

    em_large = ExecutionModel(cfg)
    res_large = em_large.compute_fill(
        raw_price=raw,
        qty=5.0,
        direction="SELL",
        bar_volume=50.0,
        apply_liquidity_cap=False,
        order_notional=10_000.0,
    )

    # For SELL, larger impact means a lower fill price (worse for the seller)
    assert res_large.fill_price < res_small.fill_price


# ── coefficient == 0 → zero impact ───────────────────────────────────────────


def test_sqrt_impact_zero_coefficient_equals_flat():
    """sqrt_impact with coefficient=0 must produce identical fill to flat model."""
    flat_cfg = _flat_cfg(random_seed=13)
    sqrt_cfg_zero = _sqrt_cfg(
        random_seed=13,
        slippage_impact_coefficient=0.0,
        slippage_adv_quote=1_000_000.0,
    )

    em_flat = ExecutionModel(flat_cfg)
    res_flat = em_flat.compute_fill(
        raw_price=100.0,
        qty=1.0,
        direction="BUY",
        bar_volume=50.0,
        apply_liquidity_cap=False,
        order_notional=500.0,
    )

    em_sqrt = ExecutionModel(sqrt_cfg_zero)
    res_sqrt = em_sqrt.compute_fill(
        raw_price=100.0,
        qty=1.0,
        direction="BUY",
        bar_volume=50.0,
        apply_liquidity_cap=False,
        order_notional=500.0,
    )

    assert res_flat.fill_price == res_sqrt.fill_price


# ── order_notional=None → flat fallback on sqrt_impact model ────────────────


def test_sqrt_impact_none_notional_equals_flat():
    """sqrt_impact + order_notional=None must fall back to flat behaviour."""
    flat_cfg = _flat_cfg(random_seed=21)
    sqrt_cfg = _sqrt_cfg(
        random_seed=21,
        slippage_impact_coefficient=0.05,
        slippage_adv_quote=1_000_000.0,
    )

    em_flat = ExecutionModel(flat_cfg)
    res_flat = em_flat.compute_fill(
        raw_price=200.0,
        qty=1.0,
        direction="BUY",
        bar_volume=100.0,
        apply_liquidity_cap=False,
        order_notional=None,
    )

    em_sqrt = ExecutionModel(sqrt_cfg)
    res_sqrt = em_sqrt.compute_fill(
        raw_price=200.0,
        qty=1.0,
        direction="BUY",
        bar_volume=100.0,
        apply_liquidity_cap=False,
        order_notional=None,
    )

    assert res_flat.fill_price == res_sqrt.fill_price


# ── impact magnitude matches manual calculation ──────────────────────────────


def test_sqrt_impact_magnitude_with_adv():
    """Verify impact = coefficient * sqrt(notional / adv) is added correctly."""
    adv = 100_000.0
    coeff = 0.02
    notional = 1_000.0
    raw = 500.0

    cfg = _sqrt_cfg(
        random_seed=3,
        slippage_impact_coefficient=coeff,
        slippage_adv_quote=adv,
    )

    em = ExecutionModel(cfg)
    result = em.compute_fill(
        raw_price=raw,
        qty=2.0,
        direction="BUY",
        bar_volume=200.0,
        apply_liquidity_cap=False,
        order_notional=notional,
    )

    # Manually derive expected: same RNG draw + computed impact
    rng = random.Random(3)
    slip = rng.uniform(cfg.slippage_rate * 0.5, cfg.slippage_rate * 1.5)
    flat_penalty = slip + cfg.spread_rate / 2.0
    impact = coeff * math.sqrt(notional / adv)
    total_penalty = flat_penalty + impact
    expected_price = raw * (1.0 + total_penalty)

    assert result.fill_price == pytest.approx(expected_price, rel=1e-12)


def test_sqrt_impact_magnitude_with_bar_volume():
    """Verify impact when adv=0: denominator = bar_volume * raw_price."""
    coeff = 0.03
    notional = 500.0
    raw = 1_000.0
    bar_vol = 50.0  # bar quote volume = 50_000

    cfg = _sqrt_cfg(
        random_seed=17,
        slippage_impact_coefficient=coeff,
        slippage_adv_quote=0.0,
    )

    em = ExecutionModel(cfg)
    result = em.compute_fill(
        raw_price=raw,
        qty=0.5,
        direction="SELL",
        bar_volume=bar_vol,
        apply_liquidity_cap=False,
        order_notional=notional,
    )

    rng = random.Random(17)
    slip = rng.uniform(cfg.slippage_rate * 0.5, cfg.slippage_rate * 1.5)
    flat_penalty = slip + cfg.spread_rate / 2.0
    denom = bar_vol * raw
    impact = coeff * math.sqrt(notional / denom)
    total_penalty = flat_penalty + impact
    expected_price = raw * (1.0 - total_penalty)

    assert result.fill_price == pytest.approx(expected_price, rel=1e-12)


# ── LMT (maker) fills are never affected ─────────────────────────────────────


def test_sqrt_impact_not_applied_to_maker_fills():
    """Market-impact must never affect LMT (maker) fills."""
    cfg = _sqrt_cfg(
        random_seed=8,
        slippage_impact_coefficient=0.5,
        slippage_adv_quote=100.0,
    )
    em = ExecutionModel(cfg)
    raw = 3_000.0
    result = em.compute_fill(
        raw_price=raw,
        qty=1.0,
        direction="BUY",
        bar_volume=100.0,
        is_maker=True,
        apply_liquidity_cap=False,
        order_notional=99_999.0,
    )
    assert result.fill_price == raw
