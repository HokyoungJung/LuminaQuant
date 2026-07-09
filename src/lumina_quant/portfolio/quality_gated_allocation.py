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
) -> dict[str, float]:
    """Cost-realistic quality score for one sleeve's gross return stream.

    Net returns are ``cost_realism.apply_cost_drag(gross_returns, turnover=...,
    regime=regime)``; ``sharpe``/``calmar`` are read from
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
    net = apply_cost_drag(
        gross, turnover=turnover_value, regime=regime, participation=participation
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


def _build_allocator(method: str) -> ERCPortfolio | HRPPortfolio:
    token = str(method or "erc").strip().lower()
    if token == "erc":
        return ERCPortfolio()
    if token == "hrp":
        return HRPPortfolio()
    raise ValueError(f"unsupported allocation method: {method!r} (expected 'erc' or 'hrp')")


def _resolve_upper(
    upper: float | Mapping[str, float] | None, ids: Sequence[str]
) -> dict[str, float] | None:
    if upper is None:
        return None
    if isinstance(upper, Mapping):
        return {sleeve_id: float(upper.get(sleeve_id, 1.0)) for sleeve_id in ids}
    cap = float(upper)
    return dict.fromkeys(ids, cap)


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
    """
    if not sleeve_returns:
        return {}
    turnovers = turnovers or {}

    quality = {
        sleeve_id: compute_sleeve_quality(
            series, turnovers.get(sleeve_id), regime=regime, participation=participation
        )
        for sleeve_id, series in sleeve_returns.items()
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
        gross = np.asarray(
            sleeve_returns[sleeve_id] if sleeve_returns[sleeve_id] is not None else [],
            dtype=np.float64,
        ).reshape(-1)
        net_series.append(
            apply_cost_drag(
                gross,
                turnover=float(turnovers.get(sleeve_id) or 0.0),
                regime=regime,
                participation=participation,
            )
        )

    min_len = min((series.size for series in net_series), default=0)
    if min_len < 2:
        # Not enough overlapping observations to estimate a covariance
        # structure; fall back to equal weight across survivors rather than
        # fail. Still deterministic (equal split, sorted id order).
        equal = 1.0 / float(len(survivors))
        raw_weights = dict.fromkeys(survivors, equal)
    else:
        # Trailing-window alignment: series of differing length are truncated
        # to the shortest survivor's length by keeping the most recent
        # observations. Callers that need calendar-exact alignment must supply
        # equal-length, co-dated series themselves.
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
            allocator = _build_allocator(method)
            raw_weights = allocator.allocate(survivors, matrix, upper=upper_map)
        if not raw_weights:
            equal = 1.0 / float(len(survivors))
            raw_weights = dict.fromkeys(survivors, equal)

    if turnover_penalty_lambda > 0.0:
        raw_weights = _turnover_tilted_weights(
            raw_weights, survivors, quality, turnover_penalty_lambda, upper
        )

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
    gross caps bounded by ``gross_cap``. When quality-gating leaves zero
    survivors, this still emits a well-formed manifest with ``children: []`` --
    the consumer's own ``manifest_children_empty`` fail-closed path (not a
    special case here) safely routes that to 100% cash.

    ``turnover_penalty_lambda`` and ``correlation_shrinkage`` are forwarded to
    :func:`allocate_quality_gated` (and the penalty to the ``sleeve_quality``
    block); both default OFF, in which case the emitted manifest is byte-identical
    to the pre-extension output.
    """
    sleeves = sleeves or {}
    source_rows = [dict(artifact) for artifact in (source_artifacts or [])]
    sleeve_returns = {sid: (spec or {}).get("returns") for sid, spec in sleeves.items()}
    turnovers = {sid: (spec or {}).get("turnover", 0.0) for sid, spec in sleeves.items()}

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
    )

    default_source_id = str(source_rows[0].get("id") or "") if len(source_rows) == 1 else ""

    children: list[dict[str, Any]] = []
    for sleeve_id in sorted(weights):
        weight = float(weights[sleeve_id])
        if weight <= 0.0:
            continue
        spec = sleeves.get(sleeve_id) or {}
        source_artifact_id = str(spec.get("source_artifact_id") or default_source_id or "")
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
            sleeve_returns.get(sid),
            turnovers.get(sid),
            regime=regime,
            participation=participation,
            turnover_penalty_lambda=turnover_penalty_lambda,
        )
        for sid in sorted(sleeves)
    }

    return {
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
