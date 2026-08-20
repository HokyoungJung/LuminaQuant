"""Portfolio-level risk scaling: the risk-free / exposure layer above the allocator.

Three-layer separation (see docs/research_note/named_quant_claude_suite_20260819.md):

1. **Allocator** (HRP / HERC / NCO / ...) decides the RELATIVE split among risky
   sleeves -- weights sum to 1 and carry no view on how much total risk to run.
2. **Risk scaling** (this module) decides the TOTAL risky exposure ``L``: final
   weights are ``L * w`` and the residual ``1 - L`` is the cash / risk-free
   sleeve.  NOTE: the simulator does NOT accrue interest on idle cash -- the
   configured ``RiskFreePolicy`` enters evaluation only as the excess-return
   benchmark (Sharpe/Sortino), so simulated cash earns 0%; any real risk-free
   yield on that residual is a live-deployment detail outside the backtest.
   When ``max_leverage > 1`` and ``L > 1`` there is NO negative-cash borrow
   representation (``cash_weight`` floors at 0, weights sum to ``L``); the
   manifest consumer's ``gross_cap`` must be raised in step or the definition
   fails closed downstream.
3. **Kelly** is the mu-dependent variant of layer 2, offered only as a gated
   ``fractional_kelly`` method because it re-introduces the expected-return
   estimation problem the mu-free allocators deliberately avoid.

Methods:
-------
``target_vol`` (primary, mu-free)
    ``L = min(max_leverage, sigma_target / sigma_hat)`` where ``sigma_hat`` is the
    per-bar standard deviation of the portfolio NET return stream implied by the
    allocator weights over the common-date aligned train+validation matrix.  The
    only estimate consumed is ``sigma_hat`` -- consistent with the lane's
    Markowitz's-curse philosophy (no ``mu`` forecast anywhere).

``fractional_kelly`` (gated, mu-dependent)
    ``L = min(max_leverage, fraction * max(0, mu_hat_excess / sigma_hat^2))``.
    ``risk_free_annual`` converts to per-bar by simple division
    (``rf / bars_per_year``), consistent with the arithmetic-return Gaussian
    approximation of the Kelly form itself.
    Because ``mu_hat`` error scales the exposure LINEARLY (estimate 8% when the
    truth is 4% and you run double the true Kelly), the spec must carry
    ``mu_evidence_confirmed: true`` -- to be set only after locked-OOS evidence
    supports the mean estimate -- otherwise this module FAILS CLOSED.

Everything is deterministic and estimation-window explicit; degenerate inputs
(zero variance, too few observations) fail closed with ``ValueError`` rather
than silently levering up.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "RiskScalingResult",
    "compute_risk_scaling",
    "kelly_mu_sensitivity",
    "resolve_risk_scaling_spec",
]

_METHODS = ("target_vol", "fractional_kelly")
# 365 (not the canonical 365.25 of indicators/annualization.py) is deliberate
# for the daily-crypto cell; the sqrt ratio is a 0.03% effect, far below any
# decision threshold, and bars_per_year is spec-declared/recorded either way.
_DEFAULT_BARS_PER_YEAR = 365.0
_DEFAULT_MAX_LEVERAGE = 1.0
_DEFAULT_MIN_OBSERVATIONS = 20
_DEFAULT_KELLY_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class RiskScalingResult:
    """Computed exposure ``L`` plus the diagnostics a manifest should record."""

    method: str
    exposure: float
    sigma_per_bar: float
    sigma_annual: float
    cash_weight: float
    bars_per_year: float
    observations: int
    diagnostics: dict[str, float] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "exposure": round(self.exposure, 10),
            "sigma_per_bar": round(self.sigma_per_bar, 10),
            "sigma_annual": round(self.sigma_annual, 10),
            "cash_weight": round(self.cash_weight, 10),
            "bars_per_year": self.bars_per_year,
            "observations": self.observations,
            "diagnostics": {key: round(value, 10) for key, value in self.diagnostics.items()},
        }


def resolve_risk_scaling_spec(spec: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate a ``risk_scaling`` block; fail closed on anything malformed.

    Returns a normalized dict (or ``None`` when ``spec`` is falsy).  The
    ``fractional_kelly`` gate lives here: without ``mu_evidence_confirmed: true``
    the spec is rejected so a mu-dependent exposure can never slip in silently.
    """
    if not spec:
        return None
    if not isinstance(spec, Mapping):
        raise ValueError("risk_scaling must be a mapping")
    method = str(spec.get("method") or "").strip().lower()
    if method not in _METHODS:
        raise ValueError(f"unsupported risk_scaling method {method!r} (expected one of {_METHODS})")
    bars_per_year = float(spec.get("bars_per_year", _DEFAULT_BARS_PER_YEAR))
    if not math.isfinite(bars_per_year) or bars_per_year <= 0.0:
        raise ValueError("risk_scaling.bars_per_year must be positive")
    max_leverage = float(spec.get("max_leverage", _DEFAULT_MAX_LEVERAGE))
    if not math.isfinite(max_leverage) or max_leverage <= 0.0:
        raise ValueError("risk_scaling.max_leverage must be positive")
    min_observations = int(spec.get("min_observations", _DEFAULT_MIN_OBSERVATIONS))
    if min_observations < 2:
        raise ValueError("risk_scaling.min_observations must be >= 2")
    out: dict[str, Any] = {
        "method": method,
        "bars_per_year": bars_per_year,
        "max_leverage": max_leverage,
        "min_observations": min_observations,
    }
    if method == "target_vol":
        per_bar = spec.get("sigma_target_per_bar")
        annual = spec.get("sigma_target_annual")
        if per_bar is not None:
            sigma_target = float(per_bar)
        elif annual is not None:
            sigma_target = float(annual) / math.sqrt(bars_per_year)
        else:
            raise ValueError(
                "risk_scaling target_vol requires sigma_target_annual or sigma_target_per_bar"
            )
        if not math.isfinite(sigma_target) or sigma_target <= 0.0:
            raise ValueError("risk_scaling sigma target must be positive")
        out["sigma_target_per_bar"] = sigma_target
        return out
    # fractional_kelly
    if spec.get("mu_evidence_confirmed") is not True:
        raise ValueError(
            "risk_scaling fractional_kelly is gated: set mu_evidence_confirmed=true only "
            "after locked-OOS evidence supports the mean estimate (mu-hat error scales "
            "the exposure linearly)"
        )
    fraction = float(spec.get("fraction", _DEFAULT_KELLY_FRACTION))
    if not math.isfinite(fraction) or not (0.0 < fraction <= 1.0):
        raise ValueError("risk_scaling.fraction must be in (0, 1]")
    risk_free_annual = float(spec.get("risk_free_annual", 0.0))
    if not math.isfinite(risk_free_annual):
        raise ValueError("risk_scaling.risk_free_annual must be finite")
    out["fraction"] = fraction
    out["risk_free_annual"] = risk_free_annual
    # Keep the confirmation flag so resolution is IDEMPOTENT: a resolved spec
    # re-entering the resolver (manifest -> allocator -> compute) must not trip
    # the gate it already passed.
    out["mu_evidence_confirmed"] = True
    return out


def _portfolio_stream(
    weights: Mapping[str, float],
    ids: Sequence[str],
    matrix: np.ndarray | None,
) -> np.ndarray:
    if matrix is None:
        raise ValueError(
            "risk_scaling requires an aligned net-return matrix; refusing to scale "
            "exposure without a volatility estimate"
        )
    panel = np.asarray(matrix, dtype=float)
    if panel.ndim != 2 or panel.shape[1] != len(ids):
        raise ValueError("risk_scaling matrix/ids shape mismatch")
    vector = np.asarray([float(weights.get(sleeve_id, 0.0)) for sleeve_id in ids], dtype=float)
    total = float(vector.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("risk_scaling requires strictly positive total allocator weight")
    # Normalize so sigma_hat describes the RELATIVE risky portfolio (sum w == 1);
    # the exposure L then scales that portfolio as one unit.
    vector = vector / total
    stream = panel @ vector
    if not np.all(np.isfinite(stream)):
        raise ValueError("risk_scaling portfolio stream contains non-finite returns")
    return stream


def compute_risk_scaling(
    weights: Mapping[str, float],
    ids: Sequence[str],
    matrix: np.ndarray | None,
    *,
    spec: Mapping[str, Any],
) -> RiskScalingResult:
    """Compute the exposure ``L`` for validated ``spec`` over the aligned matrix."""
    resolved = resolve_risk_scaling_spec(spec)
    if resolved is None:
        raise ValueError("risk_scaling spec is empty")
    stream = _portfolio_stream(weights, ids, matrix)
    observations = int(stream.size)
    if observations < int(resolved["min_observations"]):
        raise ValueError(
            f"risk_scaling needs >= {resolved['min_observations']} aligned observations "
            f"(got {observations})"
        )
    sigma = float(np.std(stream, ddof=1))
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("risk_scaling sigma estimate is degenerate (zero/non-finite)")
    bars_per_year = float(resolved["bars_per_year"])
    max_leverage = float(resolved["max_leverage"])
    method = str(resolved["method"])
    diagnostics: dict[str, float] = {}
    if method == "target_vol":
        sigma_target = float(resolved["sigma_target_per_bar"])
        exposure = min(max_leverage, sigma_target / sigma)
        diagnostics["sigma_target_per_bar"] = sigma_target
        diagnostics["sigma_target_annual"] = sigma_target * math.sqrt(bars_per_year)
    else:  # fractional_kelly (already gated in resolve)
        rf_per_bar = float(resolved["risk_free_annual"]) / bars_per_year
        mu_excess = float(np.mean(stream)) - rf_per_bar
        variance = sigma * sigma
        full_kelly = max(0.0, mu_excess / variance)
        exposure = min(max_leverage, float(resolved["fraction"]) * full_kelly)
        diagnostics["mu_excess_per_bar"] = mu_excess
        diagnostics["full_kelly_exposure"] = full_kelly
        diagnostics["fraction"] = float(resolved["fraction"])
    exposure = max(0.0, float(exposure))
    return RiskScalingResult(
        method=method,
        exposure=exposure,
        sigma_per_bar=sigma,
        sigma_annual=sigma * math.sqrt(bars_per_year),
        cash_weight=max(0.0, 1.0 - exposure),
        bars_per_year=bars_per_year,
        observations=observations,
        diagnostics=diagnostics,
    )


def kelly_mu_sensitivity(
    weights: Mapping[str, float],
    ids: Sequence[str],
    matrix: np.ndarray | None,
    *,
    fraction: float = _DEFAULT_KELLY_FRACTION,
    risk_free_annual: float = 0.0,
    bars_per_year: float = _DEFAULT_BARS_PER_YEAR,
    max_leverage: float = _DEFAULT_MAX_LEVERAGE,
    mu_shocks: Sequence[float] = (-0.5, 0.0, 0.5),
) -> dict[str, float]:
    """Fractional-Kelly exposure under multiplicative mu-hat errors.

    Returns ``{"mu_x0.5": L, "mu_x1": L, "mu_x1.5": L}``-style entries: the
    essay-style demonstration that a mu-hat error moves the Kelly exposure
    LINEARLY while the target-vol exposure is invariant to mu entirely.  This is
    a diagnostic (no gate) -- it never feeds an allocation.
    """
    stream = _portfolio_stream(weights, ids, matrix)
    if stream.size < 2:
        raise ValueError("kelly_mu_sensitivity needs >= 2 observations")
    sigma = float(np.std(stream, ddof=1))
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("kelly_mu_sensitivity sigma estimate is degenerate")
    rf_per_bar = float(risk_free_annual) / float(bars_per_year)
    mu_excess = float(np.mean(stream)) - rf_per_bar
    variance = sigma * sigma
    out: dict[str, float] = {}
    for shock in mu_shocks:
        scaled_mu = mu_excess * (1.0 + float(shock))
        exposure = min(float(max_leverage), float(fraction) * max(0.0, scaled_mu / variance))
        out[f"mu_x{1.0 + float(shock):g}"] = round(exposure, 10)
    return out
