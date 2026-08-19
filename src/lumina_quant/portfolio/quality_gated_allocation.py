"""Offline quality-gated sleeve allocator: static research manifest generator.

This module builds a deterministic, offline sleeve-allocation *research
manifest* from static return streams. It is one half of the Lane M2 meta
spine (the other half is the CLI wrapper in
``scripts/research/build_quality_gated_allocation.py``): given a fixed set of
candidate "sleeves" (each a synthetic or backtested return series + turnover +
strategy definition), it (1) scores each sleeve's cost-realistic quality, (2)
gates out sleeves with non-positive net Sharpe, (3) allocates capital across
the survivors with a risk-parity-style optimizer, and (4) emits a manifest
JSON payload that satisfies the fail-closed provenance contract already
enforced by the live
:class:`lumina_quant.strategies.artifact_portfolio_mode.ArtifactPortfolioModeStrategy`
consumer.

Theory anchors
--------------
* Equal-Risk-Contribution / risk parity: Maillard, Roncalli & Teiletche
  (2010), "The Properties of Equally Weighted Risk Contribution Portfolios".
* Hierarchical Risk Parity: Lopez de Prado (2016), "Building Diversified
  Portfolios that Outperform Out-of-Sample".

Both allocators are consumed as-is from
:mod:`lumina_quant.portfolio.optimizers_extra` (``ERCPortfolio`` /
``HRPPortfolio``); this module does not reimplement their numerics.

OFFLINE-only framing
---------------------
Everything in this module is research/offline tooling. It generates a static
JSON manifest artifact for human/CLI consumption; it is NOT imported by any
live or backtest hot path, is not registered as a strategy, and never touches
real capital. The *consumer* of the manifest
(``ArtifactPortfolioModeStrategy`` / ``manifest:<path>`` portfolio mode) and
its fail-closed gates already exist and are unmodified by this module -- this
module only has to produce a payload that satisfies that consumer's existing
checks (or, on any omission, correctly falls back to cash via the consumer's
own fail-closed logic).

Distinct from ``strategy_quality.py``
--------------------------------------
:mod:`lumina_quant.portfolio.strategy_quality` (``StrategyQualityOverlay``)
gates individual *live* signals at runtime from rolling realized performance
state. This module allocates *across sleeves*, offline, from static
deterministic quality scores computed once over a fixed historical return
series -- there is no runtime state, no live signal gating, and no shared
code path with the overlay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import numpy as np

from lumina_quant.portfolio.optimizer_core import metrics, project_simplex_with_upper_bounds
from lumina_quant.portfolio.optimizers_extra import ERCPortfolio, HRPPortfolio
from lumina_quant.research.cost_realism import DEFAULT_PARTICIPATION, CostRegime, apply_cost_drag

__all__ = [
    "REFERENCE_COST_REGIME_20BPS",
    "allocate_quality_gated",
    "build_allocation_manifest",
    "compute_sleeve_quality",
]

# A CostRegime whose round-trip cost is exactly 20bps: one_side_cost_fraction
# = taker_fee_rate + 0.5 * spread_rate + slippage_rate = 0.0004 + 0.0001 +
# 0.0005 = 0.0010 (10bps); roundtrip_cost_fraction = 2 * one_side = 0.0020
# (20bps). This is the fixed reference cost level used throughout the
# alpha-hunt data-PC promotion rule ("20bps reference cost").
REFERENCE_COST_REGIME_20BPS = CostRegime(
    name="quality_gate_20bps_reference",
    taker_fee_rate=0.0004,
    spread_rate=0.0002,
    slippage_rate=0.0005,
)

# Static provenance labels: the quality score is computed once, offline, over
# a fixed train+validation return series. No current-fold OOS bar is ever
# consulted for selection, objective, pruning, thresholding, tie-breaking,
# correlation, or sizing.
_SELECTION_INPUTS = ("train", "validation")

# Correlation-clustering threshold for the optional shrinkage-aware HRP linkage
# path; mirrors ``optimizers_extra.HRPPortfolio``'s default so the shrunk-linkage
# variant differs from the OFF path ONLY by the Ledoit-Wolf correlation shrinkage.
_HRP_CORR_THRESHOLD = 0.60


def _round(value: Any, ndigits: int = 10) -> float:
    return round(float(value), ndigits)


def _normalize_locked_oos_evaluation(value: Mapping[str, Any]) -> dict[str, int | float]:
    """Validate and canonicalize the pre-registered locked-OOS evaluation contract."""
    keys = (
        "rebalance_every_observations",
        "allocation_cost_bps",
        "periods_per_year",
    )
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"locked_oos_evaluation must contain exactly {list(keys)}")
    cadence = value["rebalance_every_observations"]
    periods = value["periods_per_year"]
    cost = value["allocation_cost_bps"]
    if isinstance(cadence, bool) or not isinstance(cadence, (int, np.integer)) or cadence <= 0:
        raise ValueError("rebalance_every_observations must be a positive integer")
    if isinstance(periods, bool) or not isinstance(periods, (int, np.integer)) or periods <= 0:
        raise ValueError("periods_per_year must be a positive integer")
    if isinstance(cost, bool) or not isinstance(cost, (int, float, np.integer, np.floating)):
        raise ValueError("allocation_cost_bps must be finite and nonnegative")
    cost_value = float(cost)
    if not np.isfinite(cost_value) or cost_value < 0.0:
        raise ValueError("allocation_cost_bps must be finite and nonnegative")
    return {
        "rebalance_every_observations": int(cadence),
        "allocation_cost_bps": cost_value,
        "periods_per_year": int(periods),
    }


def _default_optimizer_provenance() -> dict[str, Any]:
    return {
        "source": "quality_gated_allocation.compute_sleeve_quality",
        "selection_inputs": list(_SELECTION_INPUTS),
        "uses_current_fold_oos": False,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_parameter_fitting": False,
        "uses_locked_oos_for_threshold": False,
        "uses_locked_oos_for_tie_break": False,
        "uses_locked_oos_for_correlation": False,
        "uses_locked_oos_for_sizing": False,
    }


def _default_correlation_provenance() -> dict[str, Any]:
    provenance = _default_optimizer_provenance()
    provenance["source"] = "quality_gated_allocation_train_validation_correlation"
    provenance["ready"] = True
    return provenance


def compute_sleeve_quality(
    returns: Sequence[float] | np.ndarray | None,
    turnover: float | None,
    *,
    regime: CostRegime = REFERENCE_COST_REGIME_20BPS,
    participation: float = DEFAULT_PARTICIPATION,
    periods_per_year: int = 365,
    turnover_penalty_lambda: float = 0.0,
    returns_are_net: bool = False,
) -> dict[str, float]:
    """Cost-realistic quality score for one sleeve return stream.

    Gross inputs are converted with ``cost_realism.apply_cost_drag``; callers
    that already supply fee/funding/slippage-adjusted returns set
    ``returns_are_net=True`` so costs are not charged twice. ``sharpe``/``calmar`` are read from
    ``optimizer_core.metrics(net)`` (which does not compute a hit rate), so
    ``hit_rate`` is self-computed as ``mean(net > 0)``. Deterministic; gracefully
    handles ``None`` / empty ``returns`` (treated as a zero-length series, which
    ``metrics()`` scores as all-zero -- a sleeve with no data never survives the
    ``net_sharpe > 0`` quality gate).

    ``turnover_penalty_lambda`` (config-gated, default ``0.0`` = OFF) encodes the
    RPT<10bps turnover-death lesson directly into the quality score
    (Frazzini-Israel-Moskowitz 2012): when it is non-zero an extra
    ``quality_score = net_sharpe - lambda * turnover`` field is exposed. When it
    is OFF the returned mapping is byte-identical to the pre-penalty output -- no
    key is added and no arithmetic touches the existing numbers -- so the emitted
    manifest is unchanged.
    """
    gross = np.asarray(returns if returns is not None else [], dtype=np.float64).reshape(-1)
    turnover_value = float(turnover) if turnover is not None else 0.0
    net = (
        gross
        if returns_are_net
        else apply_cost_drag(
            gross, turnover=turnover_value, regime=regime, participation=participation
        )
    )
    stats = metrics(net, periods_per_year=periods_per_year)
    hit_rate = float(np.mean(net > 0.0)) if net.size > 0 else 0.0
    quality = {
        "n_obs": int(net.size),
        "turnover": turnover_value,
        "net_sharpe": _round(stats["sharpe"]),
        "net_calmar": _round(stats["calmar"]),
        "hit_rate": _round(hit_rate),
    }
    if turnover_penalty_lambda != 0.0:
        quality["quality_score"] = _round(
            quality["net_sharpe"] - float(turnover_penalty_lambda) * turnover_value
        )
    return quality


def _timestamp(value: Any, *, sleeve_id: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid return timestamp for sleeve {sleeve_id!r}")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid return timestamp for sleeve {sleeve_id!r}: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _prepare_return_series(
    sleeve_returns: Mapping[str, Sequence[float] | np.ndarray | None],
    return_timestamps: Mapping[str, Sequence[str]] | None,
) -> tuple[dict[str, np.ndarray], str, int]:
    """Validate streams and, when timestamped, align their exact UTC intersection."""
    series = {
        sleeve_id: np.asarray(values if values is not None else [], dtype=np.float64).reshape(-1)
        for sleeve_id, values in sleeve_returns.items()
    }
    if any(not np.all(np.isfinite(values)) for values in series.values()):
        raise ValueError("return streams must contain only finite values")
    active = sorted(sleeve_id for sleeve_id, values in series.items() if values.size > 0)
    if not active:
        return series, "trailing_min_length", 0

    timestamps = return_timestamps or {}
    timestamped = [sleeve_id for sleeve_id in active if sleeve_id in timestamps]
    if not timestamped:
        return series, "trailing_min_length", min(series[sleeve_id].size for sleeve_id in active)
    if len(timestamped) != len(active):
        raise ValueError("timestamped and untimestamped return streams cannot be mixed")

    parsed_by_id: dict[str, list[datetime]] = {}
    for sleeve_id in active:
        timestamp_values = timestamps[sleeve_id]
        if not isinstance(timestamp_values, Sequence) or isinstance(timestamp_values, str):
            raise ValueError(f"invalid return timestamps for sleeve {sleeve_id!r}")
        raw = list(timestamp_values)
        if len(raw) != series[sleeve_id].size:
            raise ValueError(f"return timestamp length mismatch for sleeve {sleeve_id!r}")
        parsed = [_timestamp(value, sleeve_id=sleeve_id) for value in raw]
        if any(current <= previous for previous, current in pairwise(parsed)):
            raise ValueError(f"return timestamps must be unique and increasing for {sleeve_id!r}")
        parsed_by_id[sleeve_id] = parsed

    common = sorted(set.intersection(*(set(parsed_by_id[sleeve_id]) for sleeve_id in active)))
    if len(common) < 2:
        raise ValueError("fewer than two common return timestamps")
    for sleeve_id in active:
        index = dict(zip(parsed_by_id[sleeve_id], series[sleeve_id]))
        series[sleeve_id] = np.asarray([index[value] for value in common], dtype=np.float64)
    return series, "exact_timestamp_intersection", len(common)


def _penalized_net_sharpe(score: Mapping[str, Any], turnover_penalty_lambda: float) -> float:
    """Turnover-penalized quality score ``net_sharpe - lambda * turnover``.

    Used for both the survivor gate and the weight tilt. When the penalty is OFF
    (``lambda == 0.0``) the stored ``net_sharpe`` is returned verbatim -- no
    arithmetic runs -- so the survivor set stays byte-identical to the pre-penalty
    gate ``net_sharpe > 0``.
    """
    net_sharpe = float(score["net_sharpe"])
    if turnover_penalty_lambda == 0.0:
        return net_sharpe
    return net_sharpe - float(turnover_penalty_lambda) * float(score.get("turnover", 0.0))


def _sample_correlation_and_std(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample (MLE, divisor ``T``) correlation matrix + per-column std.

    ``matrix`` is a finite ``(T, N)`` return matrix. Degenerate (zero-variance)
    columns get a zero std and a unit self-correlation with zero cross terms, so
    the result is always well-formed (diagonal exactly 1).
    """
    data = np.asarray(matrix, dtype=np.float64)
    n = int(data.shape[1]) if data.ndim == 2 else 0
    if data.ndim != 2 or data.shape[0] < 2 or n == 0:
        return np.eye(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    t = int(data.shape[0])
    demeaned = data - data.mean(axis=0, keepdims=True)
    sample_cov = (demeaned.T @ demeaned) / float(t)
    std = np.sqrt(np.clip(np.diag(sample_cov), 0.0, None))
    denom = np.outer(std, std)
    corr = np.divide(sample_cov, denom, out=np.zeros_like(sample_cov), where=denom > 0.0)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    return corr, std


def _hand_rolled_lw_correlation_shrinkage(
    matrix: np.ndarray, correlation_shrinkage: bool | float
) -> tuple[np.ndarray, float, np.ndarray]:
    """Hand-rolled Ledoit-Wolf (2004) shrinkage of the sample CORRELATION toward
    the identity target -- pure numpy, NO ``sklearn.covariance`` / scipy import.

    Returns ``(R_shrunk, intensity, std)`` with
    ``R_shrunk = (1 - d) * R + d * I`` (diagonal exactly 1, and PSD as a convex
    blend of two PSD matrices). ``correlation_shrinkage is True`` uses the
    closed-form analytic intensity
    ``d = (sum_{i!=j} (1 - r_ij^2)^2 / T) / (sum_{i!=j} r_ij^2)`` -- the asymptotic
    variance of the Pearson correlation over its off-diagonal Frobenius distance to
    the identity, clipped to ``[0, 1]``; a numeric value is used directly as a
    fixed intensity clipped to ``[0, 1]``.
    """
    corr, std = _sample_correlation_and_std(matrix)
    n = int(corr.shape[0])
    if n == 0:
        return corr, 0.0, std
    off = ~np.eye(n, dtype=bool)
    if correlation_shrinkage is True:
        t = float(np.asarray(matrix, dtype=np.float64).shape[0])
        off_sq = corr[off] ** 2
        denom = float(np.sum(off_sq))
        if denom <= 0.0 or t < 2.0:
            intensity = 0.0
        else:
            numer = float(np.sum((1.0 - off_sq) ** 2)) / t
            intensity = min(1.0, max(0.0, numer / denom))
    else:
        intensity = min(1.0, max(0.0, float(correlation_shrinkage)))
    shrunk = (1.0 - intensity) * corr + intensity * np.eye(n, dtype=np.float64)
    np.fill_diagonal(shrunk, 1.0)
    return shrunk, float(intensity), std


def _inverse_variance_split(variances: np.ndarray) -> np.ndarray:
    """Deterministic inverse-variance weights with an equal-weight fallback."""
    var = np.asarray(variances, dtype=np.float64).reshape(-1)
    n = int(var.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    positive = var > 0.0
    if not np.any(positive):
        return np.full(n, 1.0 / float(n), dtype=np.float64)
    inv = np.zeros(n, dtype=np.float64)
    inv[positive] = 1.0 / var[positive]
    total = float(inv.sum())
    if total <= 0.0:
        return np.full(n, 1.0 / float(n), dtype=np.float64)
    return inv / total


def _correlation_clusters(corr: np.ndarray, threshold: float) -> list[list[int]]:
    """Greedy correlation clustering over ``corr`` (row/col order preserved).

    Mirrors :func:`optimizer_core.cluster_by_correlation`: each index joins the
    first existing cluster containing a member correlated at least ``threshold``
    (absolute), otherwise it seeds a new cluster.
    """
    n = int(corr.shape[0])
    thr = abs(float(threshold))
    clusters: list[list[int]] = []
    for i in range(n):
        placed = False
        for cluster in clusters:
            if any(abs(float(corr[i, member])) >= thr for member in cluster):
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def _hrp_weights_with_correlation_shrinkage(
    survivors: Sequence[str],
    matrix: np.ndarray,
    *,
    correlation_shrinkage: bool | float,
    corr_threshold: float = _HRP_CORR_THRESHOLD,
) -> dict[str, float]:
    """Correlation-cluster HRP over a Ledoit-Wolf-shrunk correlation linkage.

    Mirrors :func:`optimizers_extra.hrp_weights_from_returns` (greedy correlation
    clustering -> inverse-variance within each cluster -> inverse cluster-variance
    across clusters), but the linkage correlation and the cluster covariance are
    the hand-rolled shrunk correlation, giving more stable OOS weights. Reached
    only when ``correlation_shrinkage`` is opted in; the OFF path keeps using the
    unmodified ``HRPPortfolio`` so its numerics stay byte-identical.
    """
    ids = list(survivors)
    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        return {ids[0]: 1.0}
    shrunk, _intensity, std = _hand_rolled_lw_correlation_shrinkage(matrix, correlation_shrinkage)
    if shrunk.shape != (n, n):
        return dict.fromkeys(ids, 1.0 / float(n))
    variances = std**2
    shrunk_cov = np.outer(std, std) * shrunk
    clusters = _correlation_clusters(shrunk, corr_threshold)

    weights = np.zeros(n, dtype=np.float64)
    cluster_variances: list[float] = []
    cluster_local: list[tuple[list[int], np.ndarray]] = []
    for members in clusters:
        local = _inverse_variance_split(variances[members])
        sub_cov = shrunk_cov[np.ix_(members, members)]
        cluster_variances.append(max(0.0, float(local @ sub_cov @ local)))
        cluster_local.append((members, local))

    across = _inverse_variance_split(np.asarray(cluster_variances, dtype=np.float64))
    for (members, local), alloc in zip(cluster_local, across):
        for member, share in zip(members, local):
            weights[member] = float(alloc) * float(share)

    total = float(weights.sum())
    if total <= 0.0:
        return dict.fromkeys(ids, 1.0 / float(n))
    weights = weights / total
    return {ids[index]: float(weights[index]) for index in range(n)}


def _turnover_tilted_weights(
    raw_weights: Mapping[str, float],
    survivors: Sequence[str],
    quality: Mapping[str, Mapping[str, Any]],
    turnover_penalty_lambda: float,
    upper: float | Mapping[str, float] | None,
) -> dict[str, float]:
    """Tilt risk-parity weights by the turnover-penalized quality score.

    ``tilted_i ~ raw_i * max(0, net_sharpe_i - lambda * turnover_i)``, renormalized
    across survivors (and re-projected under ``upper`` caps when supplied), so a
    higher-turnover sleeve with the same gross edge as a lower-turnover peer gets a
    strictly smaller weight. Invoked only when the penalty is ON (``lambda > 0``);
    the OFF path never calls this, so default-flag weights are byte-identical.
    """
    scores = {
        sid: max(0.0, _penalized_net_sharpe(quality[sid], turnover_penalty_lambda))
        for sid in survivors
    }
    tilted = {sid: max(0.0, float(raw_weights.get(sid, 0.0))) * scores[sid] for sid in survivors}
    total = float(sum(tilted.values()))
    if total <= 0.0:
        return dict(raw_weights)
    normalized = {sid: tilted[sid] / total for sid in survivors}
    upper_map = _resolve_upper(upper, survivors)
    if upper_map is not None:
        normalized = project_simplex_with_upper_bounds(normalized, upper=upper_map, target_sum=1.0)
    return normalized


# Dispersion floor for the family meta-momentum tilt: below this the family
# Sharpe cross-section (or an equal-weight member stream's std) carries no usable
# signal, so the tilt is inert and the base weights pass through unchanged.
_FAMILY_MOMENTUM_EPS = 1e-12


def _family_trailing_net_sharpe(stream: np.ndarray) -> float:
    """Non-annualized net Sharpe ``mean / sample-std`` of one return stream.

    The ``_series_sharpe`` idiom without the ``sqrt(periods_per_year)`` scale
    (family-vs-family z-scoring is scale-free, so annualization is a no-op that
    only adds a constant factor). Degenerate streams (fewer than 2 observations
    or ~zero dispersion) score 0.0 -- never raises.
    """
    data = np.asarray(stream, dtype=np.float64).reshape(-1)
    if data.size < 2:
        return 0.0
    mu = float(np.mean(data))
    sigma = float(np.std(data, ddof=1))
    if sigma <= _FAMILY_MOMENTUM_EPS:
        return 0.0
    return mu / sigma


def _family_meta_momentum_tilted_weights(
    base_weights: Mapping[str, float],
    survivors: Sequence[str],
    net_matrix: np.ndarray,
    families: Mapping[str, str] | None,
    *,
    window: int,
    strength: float,
    cap: float,
    min_families: int,
    upper: float | Mapping[str, float] | None,
) -> dict[str, float]:
    """Bounded recency tilt on the base weights by trailing family net Sharpe.

    Cross-sleeve FAMILY meta-momentum (Gupta-Kelly 2019 "Factor Momentum
    Everywhere"; Ehsani-Linnainmaa 2022): over the LAST ``window`` observations of
    the SAME trailing-aligned NET return ``net_matrix`` the covariance already
    consumed (so no out-of-sample bar can enter by construction), score each
    surviving-sleeve FAMILY by the non-annualized net Sharpe of its
    equal-weighted member return stream, z-score those Sharpes ACROSS families,
    and multiply each sleeve's base weight by
    ``clip(1 + strength * z_family, 1 - cap, 1 + cap)`` before renormalizing and
    re-projecting under ``upper`` caps. A recently-outperforming family is
    up-weighted (never zeroed, never levered) within the ``+/- cap`` band; an
    unmapped sleeve keeps a neutral ``1.0`` multiplier.

    Invoked ONLY when ``window > 0`` (the caller's default ``0`` never reaches
    here), so the flag-OFF allocator stays byte-identical. Degenerate inputs
    (empty survivors, mis-shaped matrix, ``window`` exceeding the trailing length,
    fewer than ``min_families`` distinct families among survivors, or zero
    cross-family Sharpe dispersion) return the base weights unchanged. Pure numpy,
    deterministic (sorted family iteration), never raises.
    """
    ids = list(survivors)
    n = len(ids)
    if n == 0:
        return dict(base_weights)
    matrix = np.asarray(net_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != n or matrix.shape[0] < 2:
        return dict(base_weights)
    win = int(window)
    if win < 2 or win > int(matrix.shape[0]):
        # ``window >= min_len`` guard: an under-length window would leak the
        # entire trailing block (or fail the min-observation floor) -- no-op.
        return dict(base_weights)

    family_of: dict[str, str | None] = {}
    for sleeve_id in ids:
        label: str | None = None
        if families:
            raw = families.get(sleeve_id)
            if raw is not None:
                label = str(raw)
        family_of[sleeve_id] = label
    distinct_families = sorted({label for label in family_of.values() if label is not None})
    if len(distinct_families) < max(1, int(min_families)):
        return dict(base_weights)

    column_of = {sleeve_id: index for index, sleeve_id in enumerate(ids)}
    trailing = matrix[-win:, :]
    family_sharpe: dict[str, float] = {}
    for label in distinct_families:
        member_columns = [column_of[sid] for sid in ids if family_of[sid] == label]
        if not member_columns:
            continue
        ew_stream = trailing[:, member_columns].mean(axis=1)
        family_sharpe[label] = _family_trailing_net_sharpe(ew_stream)
    if len(family_sharpe) < max(1, int(min_families)):
        return dict(base_weights)

    labels = sorted(family_sharpe)
    sharpe_values = np.asarray([family_sharpe[label] for label in labels], dtype=np.float64)
    mean_sharpe = float(np.mean(sharpe_values))
    std_sharpe = float(np.std(sharpe_values, ddof=1)) if sharpe_values.size > 1 else 0.0
    if std_sharpe <= _FAMILY_MOMENTUM_EPS:
        return dict(base_weights)
    z_by_family = {label: (family_sharpe[label] - mean_sharpe) / std_sharpe for label in labels}

    low = 1.0 - float(cap)
    high = 1.0 + float(cap)
    tilted: dict[str, float] = {}
    for sleeve_id in ids:
        label = family_of[sleeve_id]
        if label is None or label not in z_by_family:
            multiplier = 1.0
        else:
            multiplier = min(high, max(low, 1.0 + float(strength) * z_by_family[label]))
        tilted[sleeve_id] = max(0.0, float(base_weights.get(sleeve_id, 0.0))) * multiplier

    total = float(sum(tilted.values()))
    if total <= 0.0:
        return dict(base_weights)
    normalized = {sleeve_id: tilted[sleeve_id] / total for sleeve_id in ids}
    upper_map = _resolve_upper(upper, ids)
    if upper_map is not None:
        normalized = project_simplex_with_upper_bounds(normalized, upper=upper_map, target_sum=1.0)
    return normalized


# Opt-in hierarchical / robust allocators (2026-08-19). Every token below is
# additive: ``"erc"`` / ``"hrp"`` keep their exact legacy classes and kwargs-free
# construction, so the default path stays byte-identical.
_EXTENDED_ALLOCATORS: dict[str, str] = {
    "hrp_dendrogram": "HRPDendrogramPortfolio",
    "hrp_full": "HRPDendrogramPortfolio",
    "constrained_hrp": "HRPDendrogramPortfolio",
    "herc": "HERCPortfolio",
    "nco": "NCOPortfolio",
    "wasserstein_dro": "WassersteinDROPortfolio",
    "dro": "WassersteinDROPortfolio",
    "graph_inverse_centrality": "GraphInverseCentralityPortfolio",
    "graph": "GraphInverseCentralityPortfolio",
}
_ALLOCATOR_METHODS: tuple[str, ...] = ("erc", "hrp", *sorted(_EXTENDED_ALLOCATORS))


def _build_allocator(method: str, allocator_params: Mapping[str, Any] | None = None):
    token = str(method or "erc").strip().lower()
    if token == "erc":
        return ERCPortfolio()
    if token == "hrp":
        return HRPPortfolio()
    class_name = _EXTENDED_ALLOCATORS.get(token)
    if class_name is None:
        raise ValueError(
            f"unsupported allocation method: {method!r} (expected one of {_ALLOCATOR_METHODS})"
        )
    from lumina_quant.portfolio import hierarchical as _hier  # local: opt-in module

    cls = getattr(_hier, class_name)
    params = {str(k): v for k, v in dict(allocator_params or {}).items()}
    if token == "constrained_hrp" and (
        "lower" not in params
        or "upper_bound" not in params
        or params["lower"] is None
        or params["upper_bound"] is None
    ):
        raise ValueError("constrained_hrp requires explicit lower and upper_bound")
    return cls(**params)


def _resolve_upper(
    upper: float | Mapping[str, float] | None, ids: Sequence[str]
) -> dict[str, float] | None:
    if upper is None:
        return None
    if isinstance(upper, Mapping):
        return {sleeve_id: float(upper.get(sleeve_id, 1.0)) for sleeve_id in ids}
    cap = float(upper)
    return dict.fromkeys(ids, cap)


def _assert_train_validation_source(sleeve_id: str, spec: Mapping[str, Any]) -> None:
    for key, value in spec.items():
        token = str(key).strip().lower()
        if (token.startswith("uses_locked_oos") or token == "uses_current_fold_oos") and bool(
            value
        ):
            raise ValueError(f"locked_oos input is forbidden for sleeve {sleeve_id!r}")

    def _oos_token(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return "locked_oos" in normalized or any(part == "oos" for part in normalized.split("_"))

    def _contains_oos(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, item in value.items():
                token = str(key).strip().lower().replace("-", "_").replace(" ", "_")
                if token.startswith("uses_locked_oos") or token in {
                    "locked_oos_used_for_weights",
                    "uses_current_fold_oos",
                }:
                    if bool(item):
                        return True
                    continue
                if _contains_oos(item):
                    return True
            return False
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return any(_contains_oos(item) for item in value)
        return _oos_token(value)

    if _contains_oos(spec.get("returns_source")):
        raise ValueError(f"locked_oos input is forbidden for sleeve {sleeve_id!r}")


def _validate_materialized_data_contract(sleeves: Mapping[str, Mapping[str, Any]]) -> None:
    active = {
        sleeve_id
        for sleeve_id, spec in sleeves.items()
        if np.asarray(
            [] if (spec or {}).get("returns") is None else (spec or {}).get("returns")
        ).size
        > 0
    }
    for sleeve_id, spec in sleeves.items():
        _assert_train_validation_source(sleeve_id, spec or {})
    opt_in = any(
        any(
            key in (spec or {})
            for key in ("return_timestamps", "returns_are_net", "returns_source")
        )
        for spec in sleeves.values()
    )
    if not opt_in:
        return
    invalid_net = sorted(
        sleeve_id
        for sleeve_id in active
        if "returns_are_net" in (sleeves[sleeve_id] or {})
        and not isinstance((sleeves[sleeve_id] or {})["returns_are_net"], (bool, np.bool_))
    )
    if invalid_net:
        raise ValueError(f"returns_are_net must be boolean for materialized sleeves: {invalid_net}")
    for key in ("return_timestamps", "returns_source"):
        missing = sorted(sleeve_id for sleeve_id in active if key not in (sleeves[sleeve_id] or {}))
        if missing:
            raise ValueError(f"{key} is required for materialized sleeves: {missing}")


def allocate_quality_gated(
    sleeve_returns: Mapping[str, Sequence[float] | np.ndarray | None] | None,
    turnovers: Mapping[str, float] | None = None,
    *,
    regime: CostRegime = REFERENCE_COST_REGIME_20BPS,
    participation: float = DEFAULT_PARTICIPATION,
    method: str = "erc",
    upper: float | Mapping[str, float] | None = None,
    min_sleeves: int = 1,
    turnover_penalty_lambda: float = 0.0,
    correlation_shrinkage: bool | float | None = None,
    families: Mapping[str, str] | None = None,
    family_momentum_window: int = 0,
    family_momentum_tilt_strength: float = 0.5,
    family_momentum_tilt_cap: float = 0.30,
    min_families: int = 3,
    allocator_params: Mapping[str, Any] | None = None,
    returns_are_net: Mapping[str, bool] | None = None,
    return_timestamps: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, float]:
    """Quality-gate then risk-allocate across sleeves; returns ``id -> weight``.

    Sleeves with cost-realistic ``net_sharpe <= 0`` (see
    :func:`compute_sleeve_quality`) are dropped. Survivors are allocated via
    ``ERCPortfolio`` (``method="erc"``, default) or ``HRPPortfolio``
    (``method="hrp"``) over the *net* (post-cost) return covariance, optionally
    capped per-sleeve by ``upper`` (a single float applied to every survivor, or
    a ``{id: cap}`` mapping). Survivor ids are always processed in sorted
    (alphabetical) order before being handed to the optimizer, so the result is
    independent of the caller's mapping iteration order -- a deterministic
    tie-break. Returns ``{}`` (an all-cash allocation) when there are fewer than
    ``min_sleeves`` survivors, mirroring the conservative default of the
    downstream manifest consumer. Gracefully handles ``None``/empty
    ``sleeve_returns``/``turnovers`` and per-sleeve ``None``/empty series.
    When ``return_timestamps`` is supplied, every non-empty stream must carry a
    unique increasing timestamp vector and all quality/covariance inputs are
    restricted to the exact common UTC intersection. ``returns_are_net`` avoids
    applying the reference cost model a second time to backtester-net streams.

    Two config-gated extensions (both default OFF -> the result is byte-identical
    to the pre-extension allocator):

    * ``turnover_penalty_lambda`` (default ``0.0``): the survivor gate uses the
      turnover-penalized score ``net_sharpe - lambda * turnover`` (so a
      high-turnover, marginal-edge sleeve is gated out), and when it is ``> 0`` the
      final risk-parity weights are additionally tilted by that penalized score so a
      higher-turnover sleeve is down-weighted relative to an equal-gross-edge,
      lower-turnover peer (Frazzini-Israel-Moskowitz 2012).
    * ``correlation_shrinkage`` (default ``None``): for ``method="hrp"`` only, the
      correlation linkage is Ledoit-Wolf-shrunk toward the identity via a
      hand-rolled closed form (``True`` = analytic intensity; a float = fixed
      intensity in ``[0, 1]``) for more stable OOS weights. ``None``/``False`` keeps
      the unmodified ``HRPPortfolio`` path; it is inert for ``method="erc"``.
    * ``family_momentum_window`` (default ``0`` = OFF): when ``> 0`` a bounded
      ``+/- family_momentum_tilt_cap`` recency tilt is applied AFTER the base
      risk-parity weights and the (optional) turnover tilt, up-weighting sleeve
      FAMILIES whose equal-weighted member stream had a higher trailing
      ``family_momentum_window``-observation net Sharpe on the SAME trailing net
      matrix the covariance used (Gupta-Kelly factor momentum, train/val-only by
      construction). ``families`` maps sleeve id -> family label (unmapped sleeves
      stay neutral); the tilt no-ops below ``min_families`` distinct families or
      zero cross-family dispersion. ``family_momentum_window == 0`` leaves every
      output byte-identical.
    """
    if not sleeve_returns:
        return {}
    if str(method or "erc").strip().lower() == "constrained_hrp":
        _build_allocator(method, allocator_params)
    turnovers = turnovers or {}
    returns_are_net = returns_are_net or {}
    prepared, _alignment, _common_observations = _prepare_return_series(
        sleeve_returns, return_timestamps
    )

    quality = {
        sleeve_id: compute_sleeve_quality(
            series,
            turnovers.get(sleeve_id),
            regime=regime,
            participation=participation,
            returns_are_net=bool(returns_are_net.get(sleeve_id, False)),
            turnover_penalty_lambda=turnover_penalty_lambda,
        )
        for sleeve_id, series in prepared.items()
    }
    survivors = sorted(
        sleeve_id
        for sleeve_id, score in quality.items()
        if _penalized_net_sharpe(score, turnover_penalty_lambda) > 0.0
    )
    if len(survivors) < max(1, int(min_sleeves)):
        return {}

    net_series: list[np.ndarray] = []
    for sleeve_id in survivors:
        values = prepared[sleeve_id]
        net_series.append(
            values
            if bool(returns_are_net.get(sleeve_id, False))
            else apply_cost_drag(
                values,
                turnover=float(turnovers.get(sleeve_id) or 0.0),
                regime=regime,
                participation=participation,
            )
        )

    min_len = min((series.size for series in net_series), default=0)
    # Captured for the optional family meta-momentum tilt below; stays ``None``
    # in the covariance-less fallback so the tilt is a no-op there.
    matrix: np.ndarray | None = None
    if min_len < 2:
        # Not enough overlapping observations to estimate a covariance
        # structure; fall back to equal weight across survivors rather than
        # fail. Still deterministic (equal split, sorted id order).
        equal = 1.0 / float(len(survivors))
        raw_weights = dict.fromkeys(survivors, equal)
    else:
        # Timestamped inputs were already intersected exactly above. Legacy
        # untimestamped callers retain the historical trailing-window behavior.
        matrix = np.column_stack([series[-min_len:] for series in net_series])
        upper_map = _resolve_upper(upper, survivors)
        if (
            correlation_shrinkage not in (None, False)
            and str(method or "erc").strip().lower() == "hrp"
        ):
            # Opt-in shrinkage-aware HRP linkage; OFF path (below) is untouched.
            raw_weights = _hrp_weights_with_correlation_shrinkage(
                survivors, matrix, correlation_shrinkage=correlation_shrinkage
            )
            if raw_weights and upper_map is not None:
                raw_weights = project_simplex_with_upper_bounds(
                    raw_weights, upper=upper_map, target_sum=1.0
                )
        else:
            allocator = _build_allocator(method, allocator_params)
            raw_weights = allocator.allocate(survivors, matrix, upper=upper_map)
        if not raw_weights:
            equal = 1.0 / float(len(survivors))
            raw_weights = dict.fromkeys(survivors, equal)

    if turnover_penalty_lambda > 0.0:
        raw_weights = _turnover_tilted_weights(
            raw_weights, survivors, quality, turnover_penalty_lambda, upper
        )

    if family_momentum_window > 0 and matrix is not None:
        raw_weights = _family_meta_momentum_tilted_weights(
            raw_weights,
            survivors,
            matrix,
            families,
            window=family_momentum_window,
            strength=family_momentum_tilt_strength,
            cap=family_momentum_tilt_cap,
            min_families=min_families,
            upper=upper,
        )

    if str(method or "erc").strip().lower() == "constrained_hrp":
        from lumina_quant.portfolio import hierarchical as _hier

        params = dict(allocator_params or {})
        lo, hi = _hier._resolve_bounds(
            {"lower": params["lower"], "upper": params["upper_bound"]}, len(survivors)
        )
        upper_map = _resolve_upper(upper, survivors)
        if upper_map is not None:
            hi = np.minimum(hi, np.asarray([upper_map[sid] for sid in survivors], dtype=float))
        projected = _hier.project_box_simplex(
            [raw_weights.get(sleeve_id, 0.0) for sleeve_id in survivors], lo, hi
        )
        raw_weights = dict(zip(survivors, projected))

    return {sleeve_id: _round(raw_weights.get(sleeve_id, 0.0)) for sleeve_id in survivors}


def build_allocation_manifest(
    sleeves: Mapping[str, Mapping[str, Any]] | None,
    *,
    source_artifacts: Sequence[Mapping[str, Any]],
    regime: CostRegime = REFERENCE_COST_REGIME_20BPS,
    participation: float = DEFAULT_PARTICIPATION,
    method: str = "erc",
    upper: float | Mapping[str, float] | None = None,
    min_sleeves: int = 1,
    gross_cap: float = 1.0,
    turnover_penalty_lambda: float = 0.0,
    correlation_shrinkage: bool | float | None = None,
    families: Mapping[str, str] | None = None,
    family_momentum_window: int = 0,
    family_momentum_tilt_strength: float = 0.5,
    family_momentum_tilt_cap: float = 0.30,
    min_families: int | None = None,
    allocator_params: Mapping[str, Any] | None = None,
    locked_oos_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest the live ``ArtifactPortfolioModeStrategy`` accepts as-is.

    ``sleeves`` maps sleeve id -> ``{"returns": [...], "turnover": float,
    "strategy_class": str, "symbols": [...], "params": {...},
    "source_artifact_id": str (optional if there is exactly one entry in
    ``source_artifacts``)}``. ``source_artifacts`` must be pre-built
    ``{"id", "path", "sha256", "max_age_hours", "ready": True,
    "portfolio_ready": True}`` rows (this module does no filesystem I/O, so it
    stays pure and deterministic -- the CLI wrapper is responsible for hashing
    the referenced files and passing the result in).

    Every field the consumer's fail-closed gate inspects is populated so a
    manifest built from quality-surviving sleeves does NOT fail-close: real
    money keys are ``False`` at both the top level and per child; no forbidden
    current-fold-OOS key is ever set; ``optimizer_provenance`` /
    ``correlation_input_provenance`` are non-empty with
    ``selection_inputs=["train", "validation"]`` (and ``ready=True`` for the
    correlation provenance); each surviving child carries
    ``no_current_fold_oos_provenance=True`` and
    ``train_validation_optimizer_provenance=True``, a ``source_artifact_id``
    that reconciles against ``source_artifacts``, and per-leaf/per-netting-group
    gross caps bounded by ``gross_cap``. Legacy callers without the materialized
    return-data contract may still emit ``children: []``; opted-in research
    inputs fail before freezing when their configured survivor floors are unmet.

    ``turnover_penalty_lambda``, ``correlation_shrinkage``, and the family
    meta-momentum kwargs (``families`` / ``family_momentum_window`` / ...) are
    forwarded to :func:`allocate_quality_gated` (and the penalty to the
    ``sleeve_quality`` block); all default OFF, in which case the emitted manifest
    is byte-identical to the pre-extension output. When ``families`` is not passed
    explicitly it is derived from each sleeve spec's optional ``"family"`` key
    (unmapped sleeves stay neutral under the tilt).
    """
    sleeves = sleeves or {}
    source_rows = [dict(artifact) for artifact in (source_artifacts or [])]
    source_by_id: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        source_id = str(source.get("id") or "")
        if not source_id or source_id in source_by_id:
            raise ValueError("source_artifacts ids must be nonempty and unique")
        source_by_id[source_id] = source
    sleeve_returns = {sid: (spec or {}).get("returns") for sid, spec in sleeves.items()}
    turnovers = {sid: (spec or {}).get("turnover", 0.0) for sid, spec in sleeves.items()}
    _validate_materialized_data_contract(sleeves)
    active_ids = {
        sleeve_id
        for sleeve_id, values in sleeve_returns.items()
        if np.asarray(values if values is not None else []).size > 0
    }
    return_timestamps = {
        sleeve_id: (spec or {}).get("return_timestamps")
        for sleeve_id, spec in sleeves.items()
        if sleeve_id in active_ids and "return_timestamps" in (spec or {})
    }
    returns_are_net = {
        sleeve_id: bool((spec or {}).get("returns_are_net", False))
        for sleeve_id, spec in sleeves.items()
        if sleeve_id in active_ids
    }
    returns_sources = {
        sleeve_id: (spec or {}).get("returns_source")
        for sleeve_id, spec in sleeves.items()
        if sleeve_id in active_ids and "returns_source" in (spec or {})
    }
    opt_in_data_contract = any(
        any(
            key in (spec or {})
            for key in ("return_timestamps", "returns_are_net", "returns_source")
        )
        for spec in sleeves.values()
    )
    if opt_in_data_contract:
        for source_id, source in source_by_id.items():
            if source.get("ready") is not True or source.get("portfolio_ready") is not True:
                raise ValueError(f"source artifact {source_id!r} is not portfolio-ready")
        default_source_id = str(source_rows[0].get("id") or "") if len(source_rows) == 1 else ""
        referenced_source_ids = {
            str((spec or {}).get("source_artifact_id") or default_source_id or "")
            for spec in sleeves.values()
        }
        for source_id in sorted(referenced_source_ids):
            source = source_by_id.get(source_id)
            if source is None:
                raise ValueError(f"referenced source artifact {source_id!r} is missing")
            if source.get("ready") is not True or source.get("portfolio_ready") is not True:
                raise ValueError(f"source artifact {source_id!r} is not portfolio-ready")
    prepared_returns, alignment, common_observations = _prepare_return_series(
        sleeve_returns, return_timestamps or None
    )
    if families is None:
        derived = {
            sid: str((spec or {}).get("family"))
            for sid, spec in sleeves.items()
            if (spec or {}).get("family") is not None
        }
        families = derived or None

    weights = allocate_quality_gated(
        sleeve_returns,
        turnovers,
        regime=regime,
        participation=participation,
        method=method,
        upper=upper,
        min_sleeves=min_sleeves,
        turnover_penalty_lambda=turnover_penalty_lambda,
        correlation_shrinkage=correlation_shrinkage,
        families=families,
        family_momentum_window=family_momentum_window,
        family_momentum_tilt_strength=family_momentum_tilt_strength,
        family_momentum_tilt_cap=family_momentum_tilt_cap,
        min_families=3 if min_families is None else min_families,
        allocator_params=allocator_params,
        returns_are_net=returns_are_net,
        return_timestamps=return_timestamps or None,
    )

    if opt_in_data_contract and len(weights) < max(1, int(min_sleeves)):
        raise ValueError(
            f"quality gate left {len(weights)} sleeves; min_sleeves={min_sleeves} forbids "
            "freezing an all-cash manifest"
        )
    if opt_in_data_contract and min_families is not None:
        surviving_families = {
            str(families[sleeve_id])
            for sleeve_id in weights
            if families is not None and sleeve_id in families
        }
        if len(surviving_families) < max(1, int(min_families)):
            raise ValueError(
                f"quality gate left {len(surviving_families)} families; "
                f"min_families={min_families} forbids freezing the manifest"
            )

    default_source_id = str(source_rows[0].get("id") or "") if len(source_rows) == 1 else ""

    children: list[dict[str, Any]] = []
    for sleeve_id in sorted(weights):
        weight = float(weights[sleeve_id])
        if weight <= 0.0:
            continue
        spec = sleeves.get(sleeve_id) or {}
        source_artifact_id = str(spec.get("source_artifact_id") or default_source_id or "")
        source = source_by_id.get(source_artifact_id)
        if source is None:
            raise ValueError(
                f"sleeve {sleeve_id!r} references missing source artifact {source_artifact_id!r}"
            )
        if source.get("ready") is not True or source.get("portfolio_ready") is not True:
            raise ValueError(f"source artifact {source_artifact_id!r} is not portfolio-ready")
        children.append(
            {
                "candidate_id": str(sleeve_id),
                "name": str(spec.get("name") or sleeve_id),
                "strategy_class": str(spec.get("strategy_class") or ""),
                "symbols": [str(symbol) for symbol in list(spec.get("symbols") or [])],
                "params": dict(spec.get("params") or {}),
                "weight": weight,
                "leaf_gross": weight,
                "leaf_gross_cap": float(gross_cap),
                "netting_group": str(sleeve_id),
                "netting_group_gross_cap": float(gross_cap),
                "source_artifact_id": source_artifact_id,
                "ready": True,
                "portfolio_ready": True,
                "real_money_execution": False,
                "allow_real_money": False,
                "ready_for_real": False,
                "no_current_fold_oos_provenance": True,
                "train_validation_optimizer_provenance": True,
                "lagged_completed_shadow_optimizer_provenance": False,
                "uses_current_fold_oos": False,
                "uses_locked_oos_for_selection": False,
                "uses_locked_oos_for_objective": False,
                "uses_locked_oos_for_pruning": False,
                "uses_locked_oos_for_parameter_fitting": False,
                "uses_locked_oos_for_threshold": False,
                "uses_locked_oos_for_tie_break": False,
                "uses_locked_oos_for_correlation": False,
                "uses_locked_oos_for_sizing": False,
                "optimizer_provenance": _default_optimizer_provenance(),
                "correlation_input_provenance": _default_correlation_provenance(),
            }
        )

    active_weight = _round(sum(child["weight"] for child in children)) if children else 0.0
    sleeve_quality = {
        sid: compute_sleeve_quality(
            prepared_returns.get(sid),
            turnovers.get(sid),
            regime=regime,
            participation=participation,
            turnover_penalty_lambda=turnover_penalty_lambda,
            returns_are_net=returns_are_net.get(sid, False),
        )
        for sid in sorted(sleeves)
    }

    manifest = {
        "artifact_kind": "quality_gated_allocation_manifest",
        "real_money_execution": False,
        "allow_real_money": False,
        "ready_for_real": False,
        "uses_current_fold_oos": False,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_parameter_fitting": False,
        "uses_locked_oos_for_threshold": False,
        "uses_locked_oos_for_tie_break": False,
        "uses_locked_oos_for_correlation": False,
        "uses_locked_oos_for_sizing": False,
        "gross_cap": float(gross_cap),
        "cash_weight": max(0.0, _round(1.0 - active_weight)),
        "allocation_method": str(method),
        "optimizer_provenance": _default_optimizer_provenance(),
        "correlation_input_provenance": _default_correlation_provenance(),
        "source_artifacts": source_rows,
        "children": children,
        "sleeve_quality": sleeve_quality,
    }
    if allocator_params:
        # Opt-in provenance only: without allocator_params the manifest stays
        # byte-identical to the pinned golden.
        manifest["allocator_params"] = {str(k): v for k, v in dict(allocator_params).items()}
    if opt_in_data_contract:
        manifest["return_data_contract"] = {
            "alignment": alignment,
            "common_observations": common_observations,
            "returns_are_net": {sid: returns_are_net[sid] for sid in sorted(active_ids)},
            "returns_source": {sid: returns_sources[sid] for sid in sorted(active_ids)},
        }
    if locked_oos_evaluation is not None:
        manifest["locked_oos_evaluation"] = _normalize_locked_oos_evaluation(locked_oos_evaluation)
    return manifest
