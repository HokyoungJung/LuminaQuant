"""Advisory Sharpe confidence-interval sub-block (research-only, emit-OFF).

A block-bootstrap two-sided confidence interval for the (annualized) Sharpe
ratio.  This is an ADVISORY diagnostic surfaced only as a SEPARATE top-level
sub-object — it is never merged into the flat ``dict[str, float]`` metric payload
(``strategy_factory.research_metrics.compute_metric_summary``), so the keyset /
flat-float guard is untouched.

Determinism: the resampling uses ``numpy.random.default_rng`` seeded from config
(fixed default seed) and a circular block bootstrap (Politis & Romano, 1994)
that reuses the same block construction as ``research_metrics.spa_like_pvalue``,
preserving serial dependence.  Emission is gated by
``ResearchConfig.sharpe_ci.emit_enabled`` (default ``False``); callers get
``None`` unless they explicitly opt in.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lumina_quant.strategy_factory.research_metrics import safe_mean, safe_std


@dataclass(frozen=True, slots=True)
class SharpeConfidenceInterval:
    """Advisory bootstrap CI for the annualized Sharpe ratio.

    All fields are bare floats/ints; ``as_subobject`` returns a nested dict that
    a caller may attach under a NEW top-level key (never into the flat payload).
    """

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    bootstrap_rounds: int
    block_size: int
    sample_size: int
    periods_per_year: int
    valid_rounds: int

    def as_subobject(self) -> dict[str, Any]:
        """Return a JSON-friendly nested dict (a top-level SUB-object, not flat)."""
        return dict(asdict(self))


def _annualized_sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    sigma = safe_std(returns)
    if sigma <= 1e-12:
        return 0.0
    return (safe_mean(returns) / sigma) * math.sqrt(float(periods_per_year))


def compute_sharpe_confidence_interval(
    returns: np.ndarray,
    *,
    periods_per_year: int = 365,
    confidence_level: float = 0.95,
    bootstrap_rounds: int = 1000,
    block_size: int | None = None,
    seed: int = 20260701,
) -> SharpeConfidenceInterval:
    """Circular block-bootstrap CI for the annualized Sharpe ratio.

    ``block_size`` of ``None`` (or a non-positive value) defaults to ``n**(1/3)``,
    the standard rate for dependent data.  Returns a degenerate CI
    (lower == upper == point) when the series is too short or has no dispersion.
    """
    arr = np.asarray(returns, dtype=float)
    n = int(arr.size)
    point = _annualized_sharpe(arr, periods_per_year)
    conf = float(min(0.999999, max(0.5, confidence_level)))
    rounds = max(1, int(bootstrap_rounds))

    if block_size is None or int(block_size) <= 0:
        block_len = max(1, round(n ** (1.0 / 3.0))) if n > 0 else 1
    else:
        block_len = max(1, min(n, int(block_size))) if n > 0 else 1

    if n < 16 or safe_std(arr) <= 1e-12:
        return SharpeConfidenceInterval(
            point_estimate=float(point),
            lower=float(point),
            upper=float(point),
            confidence_level=conf,
            bootstrap_rounds=rounds,
            block_size=int(block_len),
            sample_size=n,
            periods_per_year=int(periods_per_year),
            valid_rounds=0,
        )

    rng = np.random.default_rng(int(seed))
    num_blocks = math.ceil(n / block_len)
    block_offsets = np.arange(block_len)[None, :]
    estimates: list[float] = []
    for _ in range(rounds):
        starts = rng.integers(0, n, size=num_blocks)
        offsets = (starts[:, None] + block_offsets) % n
        sample = arr[offsets.reshape(-1)[:n]]
        if safe_std(sample) <= 1e-12:
            continue
        estimates.append(_annualized_sharpe(sample, periods_per_year))

    if not estimates:
        return SharpeConfidenceInterval(
            point_estimate=float(point),
            lower=float(point),
            upper=float(point),
            confidence_level=conf,
            bootstrap_rounds=rounds,
            block_size=int(block_len),
            sample_size=n,
            periods_per_year=int(periods_per_year),
            valid_rounds=0,
        )

    boot = np.sort(np.asarray(estimates, dtype=float))
    alpha = 1.0 - conf
    lower = float(np.quantile(boot, alpha / 2.0, method="linear"))
    upper = float(np.quantile(boot, 1.0 - alpha / 2.0, method="linear"))
    return SharpeConfidenceInterval(
        point_estimate=float(point),
        lower=lower,
        upper=upper,
        confidence_level=conf,
        bootstrap_rounds=rounds,
        block_size=int(block_len),
        sample_size=n,
        periods_per_year=int(periods_per_year),
        valid_rounds=int(boot.size),
    )


def maybe_emit_sharpe_ci_subobject(
    returns: np.ndarray,
    *,
    config: Any,
    periods_per_year: int = 365,
) -> dict[str, Any] | None:
    """Return the advisory Sharpe-CI sub-object, or ``None`` when emission is OFF.

    ``config`` is a ``SharpeConfidenceIntervalConfig`` (or any object exposing the
    same attributes).  When ``emit_enabled`` is falsy this returns ``None`` and
    performs no bootstrap work — the default, byte-inert path.
    """
    if not bool(getattr(config, "emit_enabled", False)):
        return None
    block_size = int(getattr(config, "block_size", 0) or 0)
    ci = compute_sharpe_confidence_interval(
        returns,
        periods_per_year=periods_per_year,
        confidence_level=float(getattr(config, "confidence_level", 0.95)),
        bootstrap_rounds=int(getattr(config, "bootstrap_rounds", 1000)),
        block_size=block_size if block_size > 0 else None,
        seed=int(getattr(config, "seed", 20260701)),
    )
    return ci.as_subobject()


__all__ = [
    "SharpeConfidenceInterval",
    "compute_sharpe_confidence_interval",
    "maybe_emit_sharpe_ci_subobject",
]
