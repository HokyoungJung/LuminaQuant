"""Survivorship gate for research alpha candidates.

The gate answers a single question: *given the whole family of candidates that
were searched, does this candidate's edge survive a multiple-testing penalty?*

Layers
------
1. Primary gate -- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) evaluated
   at ``num_trials = N_eff`` (the eigen-free effective number of independent trials,
   see :mod:`lumina_quant.research.effective_trials`) and with
   ``variance_across_trials`` estimated *empirically* from the family's Sharpe
   cross-section -- never the single-series fallback. The raw candidate count is
   logged only, never used as the trial count.
2. Auxiliary sanity -- a single-strategy SPA / reality-check p-value (this is a
   per-candidate sanity check, NOT a family-wise error claim) and the
   probability-of-backtest-overfitting fold-instability check.

Only ``research_metrics`` primitives are reused; no scipy / sklearn / eigen calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lumina_quant.research.effective_trials import (
    effective_number_of_trials,
    raw_number_of_trials,
)
from lumina_quant.strategy_factory.research_metrics import (
    approx_pbo,
    deflated_sharpe_ratio,
    spa_like_pvalue,
)

__all__ = [
    "DEFAULT_DSR_THRESHOLD",
    "DEFAULT_PBO_THRESHOLD",
    "DEFAULT_SPA_PVALUE_THRESHOLD",
    "DEFAULT_SPA_SEED",
    "SurvivorshipVerdict",
    "empirical_variance_across_trials",
    "evaluate_survivorship_gate",
]

# Module-level policy constants. C2b owns any config-gated overrides; these are the
# in-module defaults so this lane adds no config keys.
DEFAULT_DSR_THRESHOLD = 0.95
DEFAULT_SPA_PVALUE_THRESHOLD = 0.05
DEFAULT_PBO_THRESHOLD = 0.5
DEFAULT_SPA_SEED = 12345


def empirical_variance_across_trials(family_sharpes: np.ndarray) -> float:
    """Empirical dispersion of the family's candidate Sharpe cross-section.

    Sorted into a canonical ascending order before the reduction so the sum is
    byte-reproducible regardless of the order candidates were discovered in.
    Uses the sample variance (``ddof=1``) of the finite Sharpes; returns ``0.0``
    for fewer than two finite values.
    """
    s = np.asarray(family_sharpes, dtype=float).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return 0.0
    s = np.sort(s, kind="stable")
    var = float(np.var(s, ddof=1))
    return max(0.0, var) if np.isfinite(var) else 0.0


@dataclass(frozen=True, slots=True)
class SurvivorshipVerdict:
    """Immutable outcome of the survivorship gate for one candidate."""

    passed: bool
    deflated_sharpe: float
    dsr_threshold: float
    n_eff: float
    raw_n: int
    variance_across_trials: float
    spa_pvalue: float
    spa_pvalue_threshold: float
    spa_ok: bool
    pbo: float
    pbo_threshold: float
    pbo_ok: bool
    require_spa: bool
    require_pbo: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_survivorship_gate(
    target_returns: np.ndarray,
    *,
    family_sharpes: np.ndarray,
    correlation_matrix: np.ndarray,
    dsr_threshold: float = DEFAULT_DSR_THRESHOLD,
    spa_pvalue_threshold: float = DEFAULT_SPA_PVALUE_THRESHOLD,
    pbo_threshold: float = DEFAULT_PBO_THRESHOLD,
    require_spa: bool = False,
    require_pbo: bool = False,
    spa_seed: int = DEFAULT_SPA_SEED,
) -> SurvivorshipVerdict:
    """Evaluate the survivorship gate for a single candidate return series.

    Parameters
    ----------
    target_returns:
        Per-period return series of the candidate under test.
    family_sharpes:
        Cross-section of candidate Sharpes in the searched family. Drives the
        empirical ``variance_across_trials`` fed to the DSR (never the single
        -series fallback).
    correlation_matrix:
        Candidate return-correlation matrix for the family. Drives ``N_eff`` (the
        independent-trial count) and ``raw_n`` (logged upper bound only).
    dsr_threshold / spa_pvalue_threshold / pbo_threshold:
        Decision thresholds. DSR is the primary gate; SPA/PBO are auxiliary and
        only affect the verdict when ``require_spa`` / ``require_pbo`` are set.

    Returns
    -------
    SurvivorshipVerdict
    """
    returns = np.asarray(target_returns, dtype=float).reshape(-1)

    raw_n = raw_number_of_trials(correlation_matrix)
    n_eff = effective_number_of_trials(correlation_matrix, raw_n=raw_n)
    variance = empirical_variance_across_trials(family_sharpes)

    dsr = float(
        deflated_sharpe_ratio(
            returns,
            num_trials=n_eff,
            variance_across_trials=variance,
        )
    )
    spa_p = float(spa_like_pvalue(returns, seed=spa_seed))
    pbo = float(approx_pbo(returns))

    dsr_ok = dsr >= float(dsr_threshold)
    spa_ok = spa_p <= float(spa_pvalue_threshold)
    pbo_ok = pbo <= float(pbo_threshold)

    passed = dsr_ok
    if require_spa:
        passed = passed and spa_ok
    if require_pbo:
        passed = passed and pbo_ok

    return SurvivorshipVerdict(
        passed=bool(passed),
        deflated_sharpe=dsr,
        dsr_threshold=float(dsr_threshold),
        n_eff=float(n_eff),
        raw_n=int(raw_n),
        variance_across_trials=float(variance),
        spa_pvalue=spa_p,
        spa_pvalue_threshold=float(spa_pvalue_threshold),
        spa_ok=bool(spa_ok),
        pbo=pbo,
        pbo_threshold=float(pbo_threshold),
        pbo_ok=bool(pbo_ok),
        require_spa=bool(require_spa),
        require_pbo=bool(require_pbo),
    )
