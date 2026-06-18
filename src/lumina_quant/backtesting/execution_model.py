"""Unified execution cost model — shared by backtest and live paths (Phase 4).

Encapsulates:
    fees (taker + maker) · slippage + spread · partial-fill liquidity cap ·
    LMT strict-cross fills · funding rate payments · leverage-based liquidation.

LMT fill assumptions (Phase 4, approved 2026-06-10 — see AGENTS.md §Execution):
    BUY  LMT fills when ``bar_low  < limit_price`` (strict, not ≤)
    SELL LMT fills when ``bar_high > limit_price`` (strict, not ≥)
    Fill price = ``limit_price`` exactly; commission uses ``maker_fee_rate``.
    No slippage applied to limit fills.
    The partial-fill liquidity cap (``max_bar_volume_ratio * bar_volume``) applies
    to both LMT and MKT fills.  ``TickReplayValidator`` (Phase 4.4) checks these
    rules against the raw aggTrades tape — LMT cross + fill-quantity feasibility
    (volume traded at-or-through the limit) and MKT fill price within the realised
    tick range — an independent reference, not the model re-checking its own params.

Phase 5 import gate: ``live/execution_live.py`` MUST import ``ExecutionModel`` from
this module (enforced by CI grep gate after Phase 5 completes).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecutionModelConfig:
    """Immutable parameter bundle — build via ``from_runtime()``."""

    taker_fee_rate: float
    maker_fee_rate: float
    slippage_rate: float
    spread_rate: float
    leverage: int
    margin_mode: str
    maintenance_margin_rate: float
    liquidation_buffer_rate: float
    funding_rate_per_8h: float
    funding_interval_hours: int
    random_seed: int
    max_bar_volume_ratio: float = 0.1
    slippage_impact_model: str = "flat"
    slippage_impact_coefficient: float = 0.0
    slippage_adv_quote: float = 0.0

    @classmethod
    def from_runtime(cls, rt: Any, *, mode: str = "backtest") -> ExecutionModelConfig:
        """Construct from a typed ``RuntimeConfig``.

        Parameters
        ----------
        rt:
            ``lumina_quant.configuration.schema.RuntimeConfig`` instance.
        mode:
            ``"backtest"`` — uses ``rt.backtest.{leverage,margin_mode,slippage_rate,random_seed}``.
            ``"live"``     — uses ``rt.live.exchange.{leverage,margin_mode}`` and
                             ``rt.execution.slippage_rate``; random_seed set to 42
                             (live path is deterministic via exchange, not local rng).
        """
        ex = rt.execution
        if mode == "live":
            leverage = int(rt.live.exchange.leverage)
            margin_mode = str(rt.live.exchange.margin_mode).strip().lower()
            slippage_rate = float(ex.slippage_rate)
            random_seed = 42
        else:
            leverage = int(rt.backtest.leverage)
            margin_mode = str(rt.backtest.margin_mode).strip().lower()
            slippage_rate = float(rt.backtest.slippage_rate)
            random_seed = int(rt.backtest.random_seed)
        return cls(
            taker_fee_rate=float(ex.taker_fee_rate),
            maker_fee_rate=float(ex.maker_fee_rate),
            slippage_rate=slippage_rate,
            spread_rate=float(ex.spread_rate),
            leverage=max(1, leverage),
            margin_mode=margin_mode,
            maintenance_margin_rate=float(ex.maintenance_margin_rate),
            liquidation_buffer_rate=float(ex.liquidation_buffer_rate),
            funding_rate_per_8h=float(ex.funding_rate_per_8h),
            funding_interval_hours=max(1, int(ex.funding_interval_hours)),
            random_seed=random_seed,
            max_bar_volume_ratio=float(getattr(ex, "max_bar_volume_ratio", 0.1)),
            slippage_impact_model=str(getattr(ex, "slippage_impact_model", "flat")).strip().lower(),
            slippage_impact_coefficient=float(getattr(ex, "slippage_impact_coefficient", 0.0)),
            slippage_adv_quote=float(getattr(ex, "slippage_adv_quote", 0.0)),
        )
        _impact_model = str(getattr(ex, "slippage_impact_model", "flat")).strip().lower()
        _impact_coeff = float(getattr(ex, "slippage_impact_coefficient", 0.0))
        if _impact_model == "sqrt_impact" and _impact_coeff <= 0.0:
            _LOGGER.warning(
                "slippage_impact_model='sqrt_impact' but slippage_impact_coefficient=%s <= 0 "
                "— market impact is inert (no extra slippage will be applied).",
                _impact_coeff,
            )


def _config_from_attrs(config: Any) -> ExecutionModelConfig:
    """Build ``ExecutionModelConfig`` from an uppercase-attr config object.

    Used only by ``SimulatedExecutionHandler`` and ``Portfolio`` when they receive
    a plain mock config (unit-test path).  Production code supplies a
    ``BacktestConfigView`` which carries ``._rt``; those callers use
    ``ExecutionModelConfig.from_runtime(config._rt)`` instead.
    """
    return ExecutionModelConfig(
        taker_fee_rate=float(
            getattr(config, "TAKER_FEE_RATE", getattr(config, "COMMISSION_RATE", 0.001))
        ),
        maker_fee_rate=float(getattr(config, "MAKER_FEE_RATE", 0.0002)),
        slippage_rate=float(getattr(config, "SLIPPAGE_RATE", 0.0005)),
        spread_rate=float(getattr(config, "SPREAD_RATE", 0.0002)),
        leverage=max(1, int(getattr(config, "LEVERAGE", 1))),
        margin_mode=str(getattr(config, "MARGIN_MODE", "isolated") or "isolated").strip().lower(),
        maintenance_margin_rate=float(getattr(config, "MAINTENANCE_MARGIN_RATE", 0.005)),
        liquidation_buffer_rate=float(getattr(config, "LIQUIDATION_BUFFER_RATE", 0.0)),
        funding_rate_per_8h=float(getattr(config, "FUNDING_RATE_PER_8H", 0.0)),
        funding_interval_hours=max(1, int(getattr(config, "FUNDING_INTERVAL_HOURS", 8))),
        random_seed=int(getattr(config, "RANDOM_SEED", 42)),
        max_bar_volume_ratio=float(getattr(config, "SIM_MAX_BAR_VOLUME_RATIO", 0.1)),
        slippage_impact_model=str(getattr(config, "SLIPPAGE_IMPACT_MODEL", "flat")).strip().lower(),
        slippage_impact_coefficient=float(getattr(config, "SLIPPAGE_IMPACT_COEFFICIENT", 0.0)),
        slippage_adv_quote=float(getattr(config, "SLIPPAGE_ADV_QUOTE", 0.0)),
    )


@dataclass(slots=True)
class FillResult:
    """Return value of :meth:`ExecutionModel.compute_fill`."""

    fill_price: float
    commission: float
    executed_qty: float
    unfilled_qty: float


class ExecutionModel:
    """Unified cost model shared by ``SimulatedExecutionHandler`` and ``live/execution_live.py``.

    LMT fill assumptions (Phase 4, approved 2026-06-10):
        BUY  LMT fills when ``bar_low  < limit_price`` (strict — not ≤).
        SELL LMT fills when ``bar_high > limit_price`` (strict — not ≥).
        Fill price = ``limit_price`` exactly; fee = ``maker_fee_rate``.
        No slippage applied to limit fills.
        Partial-fill cap (``max_bar_volume_ratio * bar_volume``) applies to both LMT and MKT.
        These rules are intentionally conservative: we only fill when the market clearly
        crosses the limit and give no price improvement.

    Phase 5 import gate: ``live/execution_live.py`` MUST import this class.
    """

    def __init__(self, cfg: ExecutionModelConfig) -> None:
        self.cfg = cfg
        # Seeded with random_seed to produce the same rng sequence as the legacy
        # FillModel that received SimulatedExecutionHandler.rng (same seed).
        self._rng = random.Random(cfg.random_seed)

    # ── Fills ─────────────────────────────────────────────────────────────────

    def compute_fill(
        self,
        *,
        raw_price: float,
        qty: float,
        direction: str,
        bar_volume: float,
        volatility: float = 0.0,
        is_maker: bool = False,
        apply_liquidity_cap: bool = True,
        order_notional: float | None = None,
    ) -> FillResult:
        """Simulate a single fill: (optional liquidity cap) → price → fee.

        For aggressive fills (``is_maker=False``, MKT/STOP/TP/TRAIL_STOP):
            slip = Uniform(slippage_rate*0.5, slippage_rate*1.5); doubled when vol > 0.01.
            fill_price = raw_price * (1 +/- (slip + spread/2)); fee = taker_fee_rate.
            The RNG is consumed once per call regardless of cap outcome, matching the
            legacy ``FillModel`` sequence (deterministic golden preservation).

            When ``slippage_impact_model == "sqrt_impact"`` and ``order_notional`` is
            provided (and > 0), an additional market-impact term is added to the penalty:
                participation = order_notional / denominator
                denominator   = slippage_adv_quote  (if > 0)
                                else bar_volume * raw_price  (per-bar quote volume)
                impact        = slippage_impact_coefficient * sqrt(participation)
                total penalty = base_flat_penalty + impact
            When ``order_notional`` is ``None`` or the model is ``"flat"``, the path is
            identical to the legacy flat model (byte-for-byte same RNG draw and arithmetic).

        For passive fills (``is_maker=True``, LMT):
            fill_price = raw_price exactly; fee = maker_fee_rate; RNG not consumed.

        Parameters
        ----------
        raw_price:
            Basis price — bar_open for MKT, limit_price for LMT, stop_price for STOP.
        qty:
            Requested quantity (positive float).
        direction:
            ``"BUY"`` or ``"SELL"``.
        bar_volume:
            Current bar's volume for the liquidity cap computation.
        volatility:
            Normalised bar range (high - low) / open; doubles slippage when > 0.01.
            Ignored for maker (LMT) fills.
        is_maker:
            ``True`` for LMT fills — exact price, maker fee, no slippage.
        apply_liquidity_cap:
            ``False`` disables the bar-volume cap — use for STOP/TP/TRAIL_STOP orders
            where the full quantity fills on trigger.
        order_notional:
            Optional order value in quote currency.  Only used when
            ``slippage_impact_model == "sqrt_impact"``; ignored (and defaults to
            ``None``) for the ``"flat"`` model so existing callers are unaffected.
        """
        if apply_liquidity_cap:
            max_qty = max(0.0, float(bar_volume) * self.cfg.max_bar_volume_ratio)
            executed = min(float(qty), max_qty) if max_qty > 0 else 0.0
        else:
            executed = float(qty)
        unfilled = float(qty) - executed

        if is_maker:
            # LMT fill: exact limit price, maker fee, rng NOT consumed.
            fill_price = float(raw_price)
            fee_rate = self.cfg.maker_fee_rate
        else:
            # Aggressive fill: adaptive slippage + half-spread + taker fee.
            # rng consumed unconditionally to preserve the same sequence as the
            # legacy FillModel (which always called rng.uniform even for qty=0).
            slip = self._rng.uniform(self.cfg.slippage_rate * 0.5, self.cfg.slippage_rate * 1.5)
            if float(volatility) > 0.01:
                slip *= 2.0
            penalty = slip + self.cfg.spread_rate / 2.0
            # Optional sqrt market-impact term — only active when explicitly configured.
            # The "flat" branch (default) is byte-identical to the legacy path.
            if (
                self.cfg.slippage_impact_model == "sqrt_impact"
                and order_notional is not None
                and float(order_notional) > 0.0
                and self.cfg.slippage_impact_coefficient != 0.0
            ):
                adv = self.cfg.slippage_adv_quote
                if adv > 0.0:
                    denominator = adv
                else:
                    denominator = float(bar_volume) * float(raw_price)
                if denominator > 0.0:
                    participation = float(order_notional) / denominator
                    impact = self.cfg.slippage_impact_coefficient * math.sqrt(participation)
                    penalty = penalty + impact
            # Safety clamp: a misconfigured coefficient must never produce a
            # negative fill price (SELL) or a >99% cost (BUY).
            # The "flat" model path is byte-identical: its penalty is typically
            # ~0.001-0.002 and is nowhere near the 0.99 ceiling.
            penalty = max(0.0, min(penalty, 0.99))
            if str(direction).upper() == "BUY":
                fill_price = float(raw_price) * (1.0 + penalty)
            else:
                fill_price = float(raw_price) * (1.0 - penalty)
            fee_rate = self.cfg.taker_fee_rate

        commission = fill_price * executed * fee_rate
        return FillResult(
            fill_price=fill_price,
            commission=commission,
            executed_qty=executed,
            unfilled_qty=unfilled,
        )

    def commission_for(
        self,
        *,
        fill_price: float,
        qty: float,
        is_maker: bool = False,
    ) -> float:
        """Commission for an already-known fill price/quantity — the single fee path.

        This is the one place fees are computed as ``fill_price * qty * fee_rate``.
        Use it whenever the executed price is already known (an exchange-reported
        real fill, or a forced liquidation at a computed trigger) and only the
        fee is needed — i.e. ``compute_fill`` would be wrong because it would
        re-apply slippage and move the price.

        ``is_maker`` selects ``maker_fee_rate`` (passive/LMT) vs ``taker_fee_rate``
        (aggressive/MKT/STOP/liquidation). Callers (live real-mode fills,
        liquidation accounting) must route through here rather than re-deriving
        the formula, so there is exactly one cost path (Phase 4 Principle 4).
        """
        fee_rate = self.cfg.maker_fee_rate if bool(is_maker) else self.cfg.taker_fee_rate
        return float(fill_price) * float(qty) * float(fee_rate)

    # ── Funding ───────────────────────────────────────────────────────────────

    def compute_funding_payment(
        self,
        *,
        signed_qty: float,
        price: float,
        periods: int,
        rate: float,
    ) -> float:
        """Return the funding cash flow for ``periods`` complete funding intervals.

        A positive return means the position holder **pays** (deducted from cash).
        Positive funding rate: longs pay, shorts receive.

        The ``Portfolio`` subtracts the result from ``current_holdings["cash"]`` and
        ``current_holdings["total"]`` and adds it to ``current_holdings["funding"]``.

        Parameters
        ----------
        signed_qty:
            Position size — positive = long, negative = short.
        price:
            Current mark / close price.
        periods:
            Number of complete funding intervals elapsed since last payment.
        rate:
            Funding rate per 8 h (may be dynamically resolved per bar).
        """
        if periods <= 0 or abs(float(signed_qty)) < 1e-12:
            return 0.0
        notional = abs(float(signed_qty)) * float(price)
        if notional <= 0 or abs(float(rate)) <= 1e-12:
            return 0.0
        interval_rate = float(rate) * (self.cfg.funding_interval_hours / 8.0)
        sign = 1.0 if float(signed_qty) > 0 else -1.0
        return sign * notional * interval_rate * float(periods)

    # ── Liquidation ───────────────────────────────────────────────────────────

    def liquidation_price(
        self,
        *,
        qty: float,
        entry_price: float,
    ) -> float | None:
        """Approximate isolated USDT-M liquidation price. Returns ``None`` when leverage ≤ 1.

        Long  : ``entry * (1 - 1/L + MMR + fee + buffer)``
        Short : ``entry * (1 + 1/L - MMR - fee - buffer)``
        """
        lev = max(1, int(self.cfg.leverage))
        if lev <= 1:
            return None
        mmr = self.cfg.maintenance_margin_rate
        buf = self.cfg.liquidation_buffer_rate
        fee = self.cfg.taker_fee_rate
        if float(qty) > 0:  # long
            factor = 1.0 - (1.0 / lev) + mmr + fee + buf
            factor = max(0.0, min(factor, 1.0))
        else:  # short
            factor = 1.0 + (1.0 / lev) - mmr - fee - buf
            factor = max(1.0, factor)
        return float(entry_price) * factor

    def check_liquidation(
        self,
        *,
        qty: float,
        entry_price: float,
        bar_low: float,
        bar_high: float,
        close_price: float,
    ) -> tuple[bool, float]:
        """Check whether the current bar breaches the liquidation price.

        Returns ``(breached, trigger_price)``; ``trigger_price`` is ``0.0`` when
        not breached.  The trigger price is ``bar_low / bar_high`` when the bar
        itself crosses the liquidation level, otherwise ``close_price``.
        """
        liq_price = self.liquidation_price(qty=qty, entry_price=entry_price)
        if liq_price is None:
            return False, 0.0

        if float(qty) > 0:  # long
            breached = (float(bar_low) > 0 and float(bar_low) <= liq_price) or float(
                close_price
            ) <= liq_price
            if not breached:
                return False, 0.0
            trigger = (
                float(bar_low)
                if float(bar_low) > 0 and float(bar_low) <= liq_price
                else float(close_price)
            )
        else:  # short
            breached = (float(bar_high) > 0 and float(bar_high) >= liq_price) or float(
                close_price
            ) >= liq_price
            if not breached:
                return False, 0.0
            trigger = (
                float(bar_high)
                if float(bar_high) > 0 and float(bar_high) >= liq_price
                else float(close_price)
            )

        return True, trigger


__all__ = [
    "ExecutionModel",
    "ExecutionModelConfig",
    "FillResult",
]
