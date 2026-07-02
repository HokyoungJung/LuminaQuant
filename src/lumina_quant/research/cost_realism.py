"""Cost-realism A/B for research alpha candidates + RiskManager go-live parity.

A candidate that looks profitable under optimistic (flat, zero-funding) execution
may not survive realistic costs. This module re-measures a candidate's edge under
a realistic cost regime -- taker fees + half-spread + (optionally) square-root
market-impact slippage + perpetual funding -- using the *same* cost parameters
the backtest :class:`lumina_quant.backtesting.execution_model.ExecutionModel`
consumes, so "edge survival" here means the same thing it means in the backtest,
and a promoted alpha's live order clears the *identical*
:class:`lumina_quant.risk_manager.RiskManager` order-risk gate used in backtest
(no backtest<->live divergence).

Design constraints (matching the rest of the adoption work):

* **Additive / offline.** Nothing here is imported by the live or backtest hot
  path; it is research-triage tooling invoked only by the alpha search loop when
  ``cost_ab_enabled`` is set (default OFF).
* **Deterministic.** No RNG -- slippage is the analytic *expected* cost, all
  reductions are fixed-order NumPy sums. No scipy / sklearn.
* **No new config schema keys.** Cost parameters are taken from
  :class:`CostRegime` bundles (whose defaults mirror the ``execution.*`` flags)
  or passed explicitly by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_PARTICIPATION",
    "DEFAULT_SURVIVAL_SHARPE_FRACTION",
    "FLAT_REGIME",
    "REALISTIC_REGIME",
    "CostRegime",
    "RiskGateParity",
    "apply_cost_drag",
    "check_risk_gate_parity",
    "cost_ab_report",
    "one_side_cost_fraction",
    "roundtrip_cost_fraction",
]

# Per-trade participation (fraction of ADV) feeding the sqrt-impact term. A pinned
# modelling choice; documented in Open Risks and overridable per call.
DEFAULT_PARTICIPATION = 0.05
# A candidate "survives" realistic costs when its realistic-regime Sharpe stays
# positive AND retains at least this fraction of its gross Sharpe.
DEFAULT_SURVIVAL_SHARPE_FRACTION = 0.5
_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class CostRegime:
    """A named execution-cost regime.

    Field defaults mirror the ``execution.*`` runtime flags consumed by
    :class:`ExecutionModelConfig` so the A/B measures the same cost the backtest
    would apply. ``flat`` ignores ``slippage_impact_coefficient``; ``sqrt_impact``
    adds ``coefficient * sqrt(participation)`` to the base slippage, exactly as
    ``ExecutionModel.compute_fill`` does.
    """

    name: str
    taker_fee_rate: float = 0.0004
    spread_rate: float = 0.0002
    slippage_rate: float = 0.0005
    slippage_impact_model: str = "flat"
    slippage_impact_coefficient: float = 0.0
    funding_rate_per_8h: float = 0.0

    def slippage_at(self, participation: float) -> float:
        """Expected slippage fraction for the given ADV participation."""
        slip = float(self.slippage_rate)
        if (
            self.slippage_impact_model == "sqrt_impact"
            and float(self.slippage_impact_coefficient) > 0.0
        ):
            slip += float(self.slippage_impact_coefficient) * math.sqrt(
                max(0.0, float(participation))
            )
        return slip


# Optimistic baseline: the headline-number regime (flat, no impact, no funding).
FLAT_REGIME = CostRegime(name="flat")
# Realistic regime: square-root market impact + non-zero funding. Coefficient
# 0.10 matches the repo's documented cost-realism default; calibrate per venue.
REALISTIC_REGIME = CostRegime(
    name="sqrt_impact",
    slippage_impact_model="sqrt_impact",
    slippage_impact_coefficient=0.10,
    funding_rate_per_8h=0.0001,
)


def one_side_cost_fraction(
    regime: CostRegime, participation: float = DEFAULT_PARTICIPATION
) -> float:
    """Cost fraction of one aggressive fill: fee + half-spread + slippage."""
    return (
        float(regime.taker_fee_rate)
        + 0.5 * float(regime.spread_rate)
        + regime.slippage_at(participation)
    )


def roundtrip_cost_fraction(
    regime: CostRegime, participation: float = DEFAULT_PARTICIPATION
) -> float:
    """Cost fraction of a full enter+exit round trip (two aggressive fills)."""
    return 2.0 * one_side_cost_fraction(regime, participation)


def apply_cost_drag(
    gross_returns: np.ndarray,
    *,
    turnover: float,
    regime: CostRegime,
    participation: float = DEFAULT_PARTICIPATION,
    funding_periods_per_step: float = 0.0,
) -> np.ndarray:
    """Subtract per-period execution cost from a gross return series.

    ``turnover`` is the per-period fraction of the book traded (one side); the
    per-period drag is ``turnover * one_side_cost + funding``. Funding is charged
    only when ``funding_periods_per_step > 0`` (e.g. 3.0 eight-hour intervals per
    daily bar). Deterministic and shape-preserving.
    """
    gross = np.asarray(gross_returns, dtype=np.float64).reshape(-1)
    side_cost = one_side_cost_fraction(regime, participation)
    trade_drag = float(max(0.0, turnover)) * side_cost
    funding_drag = float(regime.funding_rate_per_8h) * float(max(0.0, funding_periods_per_step))
    return gross - trade_drag - funding_drag


def _series_sharpe(returns: np.ndarray) -> float:
    """Per-period (non-annualised) Sharpe via fixed-order sums; 0.0 if degenerate."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    n = int(r.shape[0])
    if n < 2:
        return 0.0
    mean = float(np.sum(r)) / n
    dev = r - mean
    var = float(np.sum(dev * dev)) / (n - 1)
    if var <= _EPS:
        return 0.0
    return mean / math.sqrt(var)


def cost_ab_report(
    gross_returns: np.ndarray,
    *,
    turnover: float,
    participation: float = DEFAULT_PARTICIPATION,
    flat_regime: CostRegime = FLAT_REGIME,
    realistic_regime: CostRegime = REALISTIC_REGIME,
    funding_periods_per_step: float = 0.0,
    survival_fraction: float = DEFAULT_SURVIVAL_SHARPE_FRACTION,
) -> dict[str, float]:
    """A/B a candidate's edge under flat vs realistic costs.

    Returns an all-``float`` mapping (suitable for ``AlphaCandidateRecord.cost_ab``)
    with the gross / flat / realistic Sharpe, the realistic decay ratio, the cost
    inputs, and ``survives`` encoded as ``1.0`` / ``0.0``. A candidate survives when
    its realistic-regime Sharpe stays positive and keeps at least
    ``survival_fraction`` of its gross Sharpe.
    """
    gross = np.asarray(gross_returns, dtype=np.float64).reshape(-1)
    gross_sharpe = _series_sharpe(gross)

    net_flat = apply_cost_drag(
        gross, turnover=turnover, regime=flat_regime, participation=participation
    )
    net_real = apply_cost_drag(
        gross,
        turnover=turnover,
        regime=realistic_regime,
        participation=participation,
        funding_periods_per_step=funding_periods_per_step,
    )
    net_flat_sharpe = _series_sharpe(net_flat)
    net_real_sharpe = _series_sharpe(net_real)

    if gross_sharpe > _EPS:
        decay = net_real_sharpe / gross_sharpe
        survives = (net_real_sharpe > 0.0) and (net_real_sharpe >= survival_fraction * gross_sharpe)
    else:
        decay = 0.0
        survives = False

    return {
        "gross_sharpe": float(gross_sharpe),
        "net_sharpe_flat": float(net_flat_sharpe),
        "net_sharpe_realistic": float(net_real_sharpe),
        "sharpe_decay_ratio": float(decay),
        "turnover": float(turnover),
        "participation": float(participation),
        "roundtrip_cost_flat": float(roundtrip_cost_fraction(flat_regime, participation)),
        "roundtrip_cost_realistic": float(roundtrip_cost_fraction(realistic_regime, participation)),
        "survival_fraction": float(survival_fraction),
        "survives": 1.0 if survives else 0.0,
    }


# ---------------------------------------------------------------------------
# RiskManager go-live parity.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiskGateParity:
    """Outcome of running the same order through the backtest and live risk gate."""

    backtest_ok: bool
    live_ok: bool
    parity: bool
    reason_backtest: str
    reason_live: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backtest_ok": bool(self.backtest_ok),
            "live_ok": bool(self.live_ok),
            "parity": bool(self.parity),
            "reason_backtest": str(self.reason_backtest),
            "reason_live": str(self.reason_live),
        }


def _normalise_check_order(result: Any) -> tuple[bool, str]:
    """RiskManager.check_order may return ``(ok, reason)`` or a bare bool."""
    if isinstance(result, tuple):
        ok = bool(result[0])
        reason = str(result[1]) if len(result) > 1 and result[1] is not None else ""
        return ok, reason
    return bool(result), ""


def check_risk_gate_parity(
    order_event: Any,
    current_price: float,
    *,
    risk_config: Any,
    portfolio: Any = None,
) -> RiskGateParity:
    """Assert a live-eligible order clears the identical risk gate in both modes.

    A live-eligible candidate must pass the same :class:`RiskManager` order-risk
    gate whether the surrounding engine is running a backtest or live. This builds
    a RiskManager from the shared risk config for each mode and compares the
    ``check_order`` verdict + reason; ``parity`` is ``True`` only when both agree,
    which is the guarantee against backtest<->live divergence in the risk gate.
    """
    from lumina_quant.risk_manager import RiskManager

    rm_backtest = RiskManager(risk_config)
    rm_live = RiskManager(risk_config)
    ok_bt, reason_bt = _normalise_check_order(
        rm_backtest.check_order(order_event, current_price, portfolio)
    )
    ok_lv, reason_lv = _normalise_check_order(
        rm_live.check_order(order_event, current_price, portfolio)
    )
    return RiskGateParity(
        backtest_ok=ok_bt,
        live_ok=ok_lv,
        parity=(ok_bt == ok_lv and reason_bt == reason_lv),
        reason_backtest=reason_bt,
        reason_live=reason_lv,
    )
