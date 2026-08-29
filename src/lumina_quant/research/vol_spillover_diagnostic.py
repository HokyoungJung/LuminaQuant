"""V-DIAG: HAR-RV leader-spillover admission diagnostic (non-trading).

Target next-day realized variance RV(t+1) per follower symbol.  Baseline:
fixed HAR-RV OLS on the follower's own lagged log-RV 1/5/22d blocks (plus a
pre-registered EWMA reference forecast).  Candidate: the same regression plus
the pre-registered leader's lagged log-RV 1/5/22d block (pairs: BTC -> each
alt in the 10-core crypto set; XAU -> XAG/XPT/XPD).  Fit with numpy ``lstsq``
on anchored validation-forward folds (pre-registered fold grid: 8 expanding
folds with a min-train floor).  Score QLIKE (Patton-robust loss) and log-RV
MSE per fold; report per-coefficient sign stability across folds; paired
circular block bootstrap (fixed seed, block ~ ``n**(1/3)``) on the
fold-stacked QLIKE differential; Benjamini-Hochberg FDR across all
leader->follower pairs.

ADMISSION RULE (pre-registered): a pair is admitted iff median fold QLIKE
improvement >= 5% AND >= 60% of folds improved AND BH-adjusted p <= 0.05.
GLOBAL VERDICT: failure (no pair passes) KILLS all direct vol-spillover
strategies and any new multivariate-vol code, per design-brief section 5.

EXTERNAL PRIOR
--------------
Corsi (2009, J. Fin. Econometrics) HAR-RV; Patton (2011, J. Econometrics)
QLIKE robustness to noisy RV proxies; Diebold-Mariano (1995) paired
predictive-ability testing; Politis-Romano (1994) circular block bootstrap
(pattern already in repo at ``research/sharpe_ci.py``); Benjamini-Hochberg
(1995) FDR; Diebold-Yilmaz (2012) volatility spillover indices; crypto vol
connectedness (Ji et al. 2019).

EXPECTED NULL (pre-registered, verbatim)
----------------------------------------
"Leader RV blocks add no out-of-fold QLIKE improvement beyond own-history HAR:
crypto RV co-moves contemporaneously, so the LAGGED leader block is largely
subsumed by the follower's own daily lag.  Metals pairs expected to fail
outright (standing negative metals-RV evidence, graveyard #16).  Realistic
pass rate: a minority of BTC->alt pairs, possibly zero."

FALSIFYING MEASUREMENT (pre-registered, verbatim)
-------------------------------------------------
"Single number per pair: BH-adjusted one-sided bootstrap p-value on mean
paired QLIKE differential, jointly with median fold QLIKE improvement.  Pair
admitted iff improvement>=5%, >=60% folds, p<=0.05; program dead iff no pair
passes.  Test-time falsification on synthetic data: NULL generator (two
independent AR(1)-in-log-RV series) must produce ~zero admissions across
seeds; PLANTED generator (follower log-RV explicitly loading on leader's lag)
must be admitted."

SEED / BLOCK CONVENTION (mirrors ``research/sharpe_ci.py``)
-----------------------------------------------------------
``numpy.random.default_rng`` seeded from the SPEC (fixed default seed
20260820); circular block bootstrap with ``block_len = max(1,
round(n ** (1/3)))``, ``num_blocks = ceil(n / block_len)``, uniform start
indices ``rng.integers(0, n, size=num_blocks)``, modular wrap, first ``n``
samples kept.  The same fixed seed is reused for every pair (marginal
p-values need no cross-pair independence); reruns are byte-identical.

DIAGNOSTIC COLUMNS (false-admission tail guard)
-----------------------------------------------
Per pair the report carries the follower RV-series excess kurtosis
(``rolling_excess_kurtosis``) and the fold-stacked QLIKE-differential
skewness (``rolling_skewness``) so pairs whose "improvement" is driven by a
handful of fat-tail days are visibly flagged (Patton-Sheppard
loss-differential caveat).  Columns are informational, not gates.

DEGENERATE INPUTS close as ``insufficient_data`` -- missing symbols, short or
non-overlapping day ranges, non-finite RV -- and never raise.  The JSON
artifact is deterministic and byte-identical across reruns (no timestamps).

FROZEN PRE-REGISTERED DESIGN -- DEFERRED LeaderRvConditionedSizingOverlay
-------------------------------------------------------------------------
(Judge revision, documentation only -- the overlay is intentionally NOT
implemented in this change.  This block freezes its design ex-ante so any
future build cannot re-tune it against test-window PnL.)

- Wrapper meta-strategy in the exact ``VolManagedRiskOverlayStrategy`` mold
  (child instantiation, EXIT signals forwarded unscaled, ``rebalance_band``
  suppression).
- Forecast: next-day follower RV via HAR-RV PLUS the leader's lagged RV block
  (BTC->alt only at proposal time), consuming
  ``lumina_quant.indicators.har_rv``.
- Exposure map: THREE pre-registered discrete levels {1.00, 0.70, 0.40} on
  the forecast's trailing percentile, with hysteresis -- drop to 0.70 above
  the 80th percentile, restore below the 60th; drop to 0.40 above the 95th
  percentile, restore below the 85th -- and a hard 24h min-dwell per level.
- DOWN-ONLY: the multiplier never exceeds 1.0; the overlay introduces zero
  return-seeking signal (vol as sizing/conditioning, the allowed use under
  the graveyard's "range-vol directional is dead, sizing only" rule).
- FALSIFIER (data-PC, pre-registered): paired comparison of the SAME child
  sleeve wrapped three ways (unwrapped / own-vol ``VolManagedRiskOverlay`` /
  this overlay) on identical splits; kill if the median per-split net-Sharpe
  delta vs the OWN-VOL wrapper is <= 0 at the 20bps cost row, or if
  max-drawdown improvement is < 0.  The own-vol wrapper -- not the unwrapped
  sleeve -- is the null to beat.
- BUILD GATE: the overlay may only be built after V-DIAG admits >= 1
  BTC->alt pair (``sizing_overlay_build_gate_open`` in this artifact);
  V-DIAG failure auto-kills the overlay with no salvage.
"""

from __future__ import annotations

import json
import math
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import Any

import numpy as np

from lumina_quant.indicators.har_rv import (
    DEFAULT_HAR_LAGS,
    LOG_RV_EPS,
    har_design,
    har_fit,
    log_rv_transform,
)
from lumina_quant.indicators.rolling_stats import rolling_excess_kurtosis, rolling_skewness

__all__ = [
    "DEFAULT_SPEC",
    "PairDiagnostic",
    "VerdictReport",
    "bh_adjusted_pvalues",
    "evaluate_pair",
    "mse_loss",
    "paired_qlike_bootstrap_pvalue",
    "qlike_loss",
    "run_diagnostic",
]

# --------------------------------------------------------------------------- #
# pre-registered SPEC (echoed verbatim into the JSON artifact)
# --------------------------------------------------------------------------- #

_CORE_ALT_FOLLOWERS: tuple[str, ...] = (
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "TRXUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "TONUSDT",
)
_METAL_FOLLOWERS: tuple[str, ...] = ("XAGUSDT", "XPTUSDT", "XPDUSDT")

DEFAULT_SPEC: dict[str, Any] = {
    "diagnostic": "v_diag_har_rv_leader_spillover",
    "version": 1,
    "har_lags": list(DEFAULT_HAR_LAGS),
    "log_rv_eps": LOG_RV_EPS,
    "forecast_log_clip": 50.0,
    "ewma_reference_lambda": 0.94,
    "num_folds": 8,
    "min_train_rows": 150,
    "min_test_rows": 20,
    "bootstrap_rounds": 2000,
    "bootstrap_seed": 20260820,
    "bootstrap_block_len": "round(n**(1/3))",
    "qlike_improvement_floor": 0.05,
    "fold_win_rate_floor": 0.60,
    "bh_alpha": 0.05,
    "session_calendar": {
        "canonical_family": "utc_seven_day_epoch_days",
        "expected_day_step": 1,
        "members": ["BTCUSDT", *_CORE_ALT_FOLLOWERS, "XAUUSDT", *_METAL_FOLLOWERS],
    },
    "pairs": [["BTCUSDT", follower] for follower in _CORE_ALT_FOLLOWERS]
    + [["XAUUSDT", follower] for follower in _METAL_FOLLOWERS],
}
_CANONICAL_SPEC_BYTES = json.dumps(
    DEFAULT_SPEC,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")

_COEF_NAMES_BASE: tuple[str, ...] = ("const", "own_d", "own_w", "own_m")
_COEF_NAMES_LEADER: tuple[str, ...] = ("leader_d", "leader_w", "leader_m")
_LINEAGE_VERSION = "vdiag-input-v3"
_CLI_LOADER_VERSION = "vdiag-cli-v3"
_ALLOWED_CLI_LOADERS = {
    "explicit_epoch_day": "_load_rv_csv",
    "canonical_intraday": "_load_closes_csv",
    "canonical_intraday_series_dir": "_load_series_dir",
}
_AUTHORITY_SEAL = object()


class _DiagnosticAuthority:
    """Unserializable capability issued after the CLI has sealed its input."""

    __slots__ = ("_initialized", "_input_digest", "_lineage", "_seal")

    def __init__(self, seal: object, lineage: dict[str, str], input_digest: str) -> None:
        if seal is not _AUTHORITY_SEAL:
            raise TypeError("diagnostic authority is minted by the CLI loader")
        object.__setattr__(self, "_seal", seal)
        object.__setattr__(self, "_lineage", MappingProxyType(dict(lineage)))
        object.__setattr__(self, "_input_digest", input_digest)
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError("diagnostic authority is immutable")
        object.__setattr__(self, name, value)


def _freeze(value: Any) -> Any:
    """Recursively freeze report evidence so nested mutation cannot alter it."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return a JSON-compatible detached copy of frozen evidence."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def qlike_loss(realized: Any, forecast: Any, *, eps: float = LOG_RV_EPS) -> np.ndarray | None:
    """Patton (2011) QLIKE loss ``sigma2/h - log(sigma2/h) - 1``, elementwise.

    Both inputs are floored at ``eps``; ``None`` on shape mismatch or
    non-finite input.  Never raises.
    """
    try:
        sigma2 = np.maximum(np.asarray(realized, dtype=float), eps)
        h = np.maximum(np.asarray(forecast, dtype=float), eps)
    except Exception:
        return None
    if sigma2.shape != h.shape:
        return None
    if not (np.all(np.isfinite(sigma2)) and np.all(np.isfinite(h))):
        return None
    ratio = sigma2 / h
    loss = ratio - np.log(ratio) - 1.0
    if not np.all(np.isfinite(loss)):
        return None
    return loss


def mse_loss(log_realized: Any, log_forecast: Any) -> np.ndarray | None:
    """Squared error on the log-RV scale, elementwise; ``None`` fail-closed."""
    try:
        actual = np.asarray(log_realized, dtype=float)
        predicted = np.asarray(log_forecast, dtype=float)
    except Exception:
        return None
    if actual.shape != predicted.shape:
        return None
    if not (np.all(np.isfinite(actual)) and np.all(np.isfinite(predicted))):
        return None
    diff = actual - predicted
    return diff * diff


# --------------------------------------------------------------------------- #
# inference helpers
# --------------------------------------------------------------------------- #
def paired_qlike_bootstrap_pvalue(
    diffs: Any,
    *,
    rounds: int,
    seed: int,
    block_size: int | None = None,
) -> float | None:
    """One-sided circular-block-bootstrap p-value on the mean differential.

    ``diffs`` are per-observation paired loss differentials (baseline minus
    candidate; positive = candidate improvement).  H0: mean <= 0.  The block
    construction mirrors ``research/sharpe_ci.py``: ``default_rng(seed)``,
    ``block_len = max(1, round(n ** (1/3)))`` when ``block_size`` is unset,
    ``num_blocks = ceil(n / block_len)``, circular wrap, first ``n`` kept.
    Resampling is done on the CENTERED differentials; the p-value is
    ``(1 + #{boot_mean >= observed_mean}) / (rounds + 1)``.

    Returns ``None`` on fewer than 16 observations or non-finite input;
    degenerate zero-dispersion differentials resolve in closed form.  Never
    raises.
    """
    try:
        arr = np.asarray(list(diffs), dtype=float)
    except Exception:
        return None
    n = int(arr.size)
    if n < 16 or not np.all(np.isfinite(arr)):
        return None
    observed_mean = float(np.mean(arr))
    if float(np.std(arr)) <= 1e-18:
        return 0.0 if observed_mean > 0.0 else 1.0

    if block_size is None or int(block_size) <= 0:
        block_len = max(1, round(n ** (1.0 / 3.0)))
    else:
        block_len = max(1, min(n, int(block_size)))
    num_blocks = math.ceil(n / block_len)
    block_offsets = np.arange(block_len)[None, :]
    centered = arr - observed_mean

    rng = np.random.default_rng(int(seed))
    rounds_val = max(1, int(rounds))
    hits = 0
    for _ in range(rounds_val):
        starts = rng.integers(0, n, size=num_blocks)
        offsets = (starts[:, None] + block_offsets) % n
        sample = centered[offsets.reshape(-1)[:n]]
        if float(np.mean(sample)) >= observed_mean:
            hits += 1
    return float((1 + hits) / (rounds_val + 1))


def bh_adjusted_pvalues(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values, aligned to input order.

    Standard monotone BH adjustment capped at 1.0; empty input yields an
    empty list.  Never raises on finite input.
    """
    count = len(pvalues)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda idx: float(pvalues[idx]))
    adjusted = [0.0] * count
    running_min = 1.0
    for rank in range(count - 1, -1, -1):
        idx = order[rank]
        candidate = float(pvalues[idx]) * count / float(rank + 1)
        running_min = min(running_min, candidate)
        adjusted[idx] = min(1.0, running_min)
    return adjusted


def _fold_bounds(
    rows: int,
    *,
    num_folds: int,
    min_train: int,
    min_test: int,
) -> list[tuple[int, int]] | None:
    """Anchored validation-forward fold bounds ``[(test_start, test_end)]``.

    The test region ``[min_train, rows)`` is split into ``num_folds``
    contiguous blocks (earlier blocks absorb the remainder, deterministic).
    ``None`` when the region cannot host ``num_folds * min_test`` rows.
    """
    region = rows - min_train
    if num_folds < 1 or region < num_folds * min_test:
        return None
    base = region // num_folds
    remainder = region % num_folds
    bounds: list[tuple[int, int]] = []
    cursor = min_train
    for fold in range(num_folds):
        width = base + (1 if fold < remainder else 0)
        bounds.append((cursor, cursor + width))
        cursor += width
    return bounds


def _sign_stability(coef_by_fold: list[np.ndarray], names: tuple[str, ...]) -> dict[str, Any]:
    """Per-coefficient majority sign and cross-fold sign-agreement fraction."""
    table: dict[str, Any] = {}
    folds = len(coef_by_fold)
    for pos, name in enumerate(names):
        signs = [1 if coefs[pos] >= 0.0 else -1 for coefs in coef_by_fold]
        positive = sum(1 for sign in signs if sign > 0)
        majority = 1 if positive * 2 >= folds else -1
        agree = sum(1 for sign in signs if sign == majority)
        table[name] = {
            "majority_sign": float(majority),
            "stability": float(agree / folds) if folds else 0.0,
        }
    return table


def _validated_spec(
    overrides: Mapping[str, Any] | None,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> tuple[dict[str, Any], bool, bool]:
    """Return a JSON snapshot plus validity and canonical-admission flags."""
    canonical = json.loads(_CANONICAL_SPEC_BYTES)
    merged = json.loads(_CANONICAL_SPEC_BYTES)
    valid = True
    if overrides is not None:
        if not isinstance(overrides, Mapping) or not set(overrides) <= set(canonical):
            valid = False
        else:
            try:
                normalized = json.loads(json.dumps(dict(overrides), allow_nan=False))
                merged.update(normalized)
            except TypeError, ValueError:
                valid = False
    if pairs is not None:
        merged["pairs"] = []
        try:
            pair_items = list(pairs)
            if not pair_items:
                valid = False
            for item in pair_items:
                if (
                    not isinstance(item, (list, tuple))
                    or len(item) != 2
                    or type(item[0]) is not str
                    or not item[0]
                    or type(item[1]) is not str
                    or not item[1]
                ):
                    valid = False
                    break
                merged["pairs"].append([item[0], item[1]])
            if not valid:
                merged["pairs"] = []
        except TypeError, ValueError:
            valid = False
    try:
        integer_fields = (
            "version",
            "num_folds",
            "min_train_rows",
            "min_test_rows",
            "bootstrap_rounds",
            "bootstrap_seed",
        )
        if any(
            type(merged[field]) is not int
            or merged[field] < (0 if field == "bootstrap_seed" else 1)
            for field in integer_fields
        ):
            valid = False
        lags = tuple(sorted({int(value) for value in merged["har_lags"]}))
        if lags != tuple(DEFAULT_HAR_LAGS):
            valid = False
        finite_positive = ("log_rv_eps", "forecast_log_clip")
        if any(
            not math.isfinite(float(merged[field])) or float(merged[field]) <= 0.0
            for field in finite_positive
        ):
            valid = False
        lam = float(merged["ewma_reference_lambda"])
        if not math.isfinite(lam) or not 0.0 < lam < 1.0:
            valid = False
        for field in (
            "qlike_improvement_floor",
            "fold_win_rate_floor",
            "bh_alpha",
        ):
            value = float(merged[field])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                valid = False
        normalized_pairs = merged["pairs"]
        if (
            type(normalized_pairs) is not list
            or not normalized_pairs
            or any(
                type(item) is not list
                or len(item) != 2
                or any(type(value) is not str or not value for value in item)
                for item in normalized_pairs
            )
            or len({tuple(item) for item in normalized_pairs}) != len(normalized_pairs)
        ):
            valid = False
    except KeyError, TypeError, ValueError, OverflowError:
        valid = False
    authoritative = valid and merged == canonical
    merged["canonical_admission_eligible"] = authoritative
    return merged, valid, authoritative


# --------------------------------------------------------------------------- #
# per-pair evaluation
# --------------------------------------------------------------------------- #
def evaluate_pair(
    follower_rv: Any,
    leader_rv: Any,
    *,
    spec: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Evaluate one leader->follower pair on ALIGNED daily RV arrays.

    Returns the full fold-level artifact dict (fold forecasts included, for
    the no-lookahead test) or ``None`` when the pair closes as
    ``insufficient_data``.  Admission fields except the BH adjustment are
    populated (BH is a cross-pair, whole-search step).  Never raises.
    """
    merged, valid_spec, _authoritative = _validated_spec(spec)
    if not valid_spec:
        return None
    # Normalize the lag spec EXACTLY as ``har_design`` does (sorted unique
    # ints) and fail closed BEFORE building any design: the coefficient
    # labels (``_COEF_NAMES_BASE + _COEF_NAMES_LEADER``) and the leader-block
    # column split below are only valid for exactly three lags.  Fewer lags
    # would raise inside ``_sign_stability`` (never-raise breach); more lags
    # would silently mislabel own-history columns as leader coefficients.
    try:
        lag_list = sorted({int(k) for k in merged["har_lags"]})
    except Exception:
        return None
    if tuple(lag_list) != tuple(DEFAULT_HAR_LAGS):
        return None
    lags = tuple(lag_list)
    eps = float(merged["log_rv_eps"])
    clip = float(merged["forecast_log_clip"])
    num_folds = int(merged["num_folds"])
    min_train = int(merged["min_train_rows"])
    min_test = int(merged["min_test_rows"])

    log_f = log_rv_transform(follower_rv, eps=eps)
    log_l = log_rv_transform(leader_rv, eps=eps)
    if log_f is None or log_l is None or log_f.size != log_l.size:
        return None
    built_f = har_design(log_f, lags=lags)
    built_l = har_design(log_l, lags=lags)
    if built_f is None or built_l is None:
        return None
    design_own, target = built_f
    design_leader_blocks = built_l[0][:, 1:]  # drop the leader intercept column
    design_candidate = np.column_stack([design_own, design_leader_blocks])
    rows = int(target.size)
    bounds = _fold_bounds(rows, num_folds=num_folds, min_train=min_train, min_test=min_test)
    if bounds is None:
        return None

    # Pre-registered EWMA reference forecast over the follower RV series.
    lam = float(merged["ewma_reference_lambda"])
    rv_f = np.exp(log_f)  # eps-floored variance scale
    ewma = np.empty_like(rv_f)
    ewma[0] = rv_f[0]
    for idx in range(1, rv_f.size):
        ewma[idx] = lam * ewma[idx - 1] + (1.0 - lam) * rv_f[idx - 1]
    max_lag = max(lags)
    ewma_rows = ewma[max_lag:]

    realized = np.exp(target)
    folds: list[dict[str, Any]] = []
    coef_by_fold: list[np.ndarray] = []
    log_floor = math.log(eps)
    for test_start, test_end in bounds:
        coef_base = har_fit(design_own[:test_start], target[:test_start])
        coef_cand = har_fit(design_candidate[:test_start], target[:test_start])
        if coef_base is None or coef_cand is None:
            return None
        pred_base = np.clip(design_own[test_start:test_end] @ coef_base, log_floor, clip)
        pred_cand = np.clip(design_candidate[test_start:test_end] @ coef_cand, log_floor, clip)
        sigma2 = realized[test_start:test_end]
        qlike_base = qlike_loss(sigma2, np.exp(pred_base), eps=eps)
        qlike_cand = qlike_loss(sigma2, np.exp(pred_cand), eps=eps)
        mse_base = mse_loss(target[test_start:test_end], pred_base)
        mse_cand = mse_loss(target[test_start:test_end], pred_cand)
        qlike_ref = qlike_loss(sigma2, ewma_rows[test_start:test_end], eps=eps)
        if qlike_base is None or qlike_cand is None or mse_base is None or mse_cand is None:
            return None
        coef_by_fold.append(coef_cand)
        folds.append(
            {
                "test_start": test_start,
                "test_end": test_end,
                "forecast_baseline": pred_base,
                "forecast_candidate": pred_cand,
                "qlike_baseline": qlike_base,
                "qlike_candidate": qlike_cand,
                "qlike_ewma_reference": qlike_ref,
                "mse_baseline": mse_base,
                "mse_candidate": mse_cand,
            }
        )

    qlike_improvements: list[float] = []
    mse_improvements: list[float] = []
    for fold in folds:
        base_mean = float(np.mean(fold["qlike_baseline"]))
        cand_mean = float(np.mean(fold["qlike_candidate"]))
        qlike_improvements.append(1.0 - cand_mean / base_mean if base_mean > 0.0 else 0.0)
        mse_base_mean = float(np.mean(fold["mse_baseline"]))
        mse_cand_mean = float(np.mean(fold["mse_candidate"]))
        mse_improvements.append(1.0 - mse_cand_mean / mse_base_mean if mse_base_mean > 0.0 else 0.0)

    stacked_base = np.concatenate([fold["qlike_baseline"] for fold in folds])
    stacked_cand = np.concatenate([fold["qlike_candidate"] for fold in folds])
    ref_blocks = [
        fold["qlike_ewma_reference"] for fold in folds if fold["qlike_ewma_reference"] is not None
    ]
    stacked_ref = np.concatenate(ref_blocks) if ref_blocks else np.asarray([], dtype=float)
    diffs = stacked_base - stacked_cand
    pvalue = paired_qlike_bootstrap_pvalue(
        diffs,
        rounds=int(merged["bootstrap_rounds"]),
        seed=int(merged["bootstrap_seed"]),
    )
    if pvalue is None:
        return None

    wins = sum(1 for improvement in qlike_improvements if improvement > 0.0)
    coef_names = _COEF_NAMES_BASE + _COEF_NAMES_LEADER
    return {
        "n_design_rows": rows,
        "folds": folds,
        "fold_qlike_improvements": qlike_improvements,
        "fold_mse_improvements": mse_improvements,
        "median_qlike_improvement": float(np.median(np.asarray(qlike_improvements))),
        "median_mse_improvement": float(np.median(np.asarray(mse_improvements))),
        "fold_win_rate": float(wins / len(folds)),
        "mean_qlike_baseline": float(np.mean(stacked_base)),
        "mean_qlike_candidate": float(np.mean(stacked_cand)),
        "mean_qlike_ewma_reference": float(np.mean(stacked_ref)) if stacked_ref.size else None,
        "bootstrap_pvalue": float(pvalue),
        "coefficient_sign_stability": _sign_stability(coef_by_fold, coef_names),
        "rv_excess_kurtosis": rolling_excess_kurtosis(np.exp(log_f).tolist()),
        "qlike_diff_skewness": rolling_skewness(diffs.tolist()),
    }


# --------------------------------------------------------------------------- #
# report dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PairDiagnostic:
    """Per-pair V-DIAG outcome (JSON-friendly scalars only)."""

    leader: str
    follower: str
    status: str  # "evaluated" | "insufficient_data"
    n_shared_days: int
    n_design_rows: int
    fold_qlike_improvements: tuple[float, ...]
    median_qlike_improvement: float | None
    median_mse_improvement: float | None
    fold_win_rate: float | None
    mean_qlike_baseline: float | None
    mean_qlike_candidate: float | None
    mean_qlike_ewma_reference: float | None
    bootstrap_pvalue: float | None
    bh_adjusted_pvalue: float | None
    coefficient_sign_stability: dict[str, Any] = field(default_factory=dict)
    rv_excess_kurtosis: float | None = None
    qlike_diff_skewness: float | None = None
    admitted: bool = False
    shared_day_start: int | None = None
    shared_day_end: int | None = None
    fold_day_bounds: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (lists for tuples, plain scalars)."""
        return {
            "leader": self.leader,
            "follower": self.follower,
            "status": self.status,
            "n_shared_days": int(self.n_shared_days),
            "n_design_rows": int(self.n_design_rows),
            "fold_qlike_improvements": list(self.fold_qlike_improvements),
            "median_qlike_improvement": self.median_qlike_improvement,
            "median_mse_improvement": self.median_mse_improvement,
            "fold_win_rate": self.fold_win_rate,
            "mean_qlike_baseline": self.mean_qlike_baseline,
            "mean_qlike_candidate": self.mean_qlike_candidate,
            "mean_qlike_ewma_reference": self.mean_qlike_ewma_reference,
            "bootstrap_pvalue": self.bootstrap_pvalue,
            "bh_adjusted_pvalue": self.bh_adjusted_pvalue,
            "coefficient_sign_stability": _thaw(self.coefficient_sign_stability),
            "rv_excess_kurtosis": self.rv_excess_kurtosis,
            "qlike_diff_skewness": self.qlike_diff_skewness,
            "admitted": bool(self.admitted),
            "shared_day_start": self.shared_day_start,
            "shared_day_end": self.shared_day_end,
            "fold_day_bounds": [list(bounds) for bounds in self.fold_day_bounds],
        }


@dataclass(frozen=True, slots=True)
class VerdictReport:
    """Whole-search V-DIAG verdict artifact (deterministic, timestamp-free)."""

    spec: Mapping[str, Any]
    pairs: tuple[PairDiagnostic, ...]
    evaluated_pairs: int
    admitted_pairs: tuple[str, ...]
    insufficient_pairs: tuple[str, ...]
    program_verdict: str  # "admitted" | "dead" | "insufficient_data"
    sizing_overlay_build_gate_open: bool
    lineage: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the nested JSON-friendly artifact payload."""
        return {
            "spec": _thaw(self.spec),
            "pairs": [pair.to_dict() for pair in self.pairs],
            "evaluated_pairs": int(self.evaluated_pairs),
            "admitted_pairs": list(self.admitted_pairs),
            "insufficient_pairs": list(self.insufficient_pairs),
            "program_verdict": self.program_verdict,
            "sizing_overlay_build_gate_open": bool(self.sizing_overlay_build_gate_open),
            "lineage": _thaw(self.lineage),
            "kill_semantics": (
                "program_verdict=dead KILLS all direct vol-spillover strategies "
                "and any new multivariate-vol code (design-brief section 5)"
            ),
        }

    def to_json(self) -> str:
        """Deterministic byte-stable JSON encoding (sorted keys, no times)."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"


# --------------------------------------------------------------------------- #
# whole-search runner
# --------------------------------------------------------------------------- #
def _normalize_symbol_series(payload: Any) -> tuple[dict[int, float], bool, str]:
    """Validate one symbol's RV input without coercion or silent row loss.

    Only explicitly day-keyed mappings or ``(days, values)`` pairs are
    authoritative.  Plain sequences are retained for diagnostic smoke runs,
    but are labelled nonauthoritative and cannot open the canonical gate.
    """
    out: dict[int, float] = {}

    def put(day: Any, value: Any) -> bool:
        if type(day) is not int or type(value) not in (int, float):
            return False
        rv = float(value)
        if not math.isfinite(rv) or rv < 0.0 or day in out:
            return False
        out[day] = rv
        return True

    try:
        if isinstance(payload, Mapping):
            for day, value in payload.items():
                if not put(day, value):
                    return {}, False, "invalid_day_or_value"
            return out, True, "explicit_day_mapping"
        if isinstance(payload, tuple) and len(payload) == 2:
            days, values = list(payload[0]), list(payload[1])
            if len(days) != len(values):
                return {}, False, "day_value_length_mismatch"
            for day, value in zip(days, values, strict=True):
                if not put(day, value):
                    return {}, False, "invalid_day_or_value"
            return out, True, "explicit_day_sequence"
        values = list(payload)
        for day, value in enumerate(values):
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                return {}, False, "invalid_plain_sequence"
            out[day] = float(value)
        return out, False, "plain_sequence_nonauthoritative"
    except TypeError, ValueError:
        return {}, False, "unreadable_payload"


def _normalized_input_digest(normalized: Mapping[str, Mapping[int, float]]) -> str:
    """Hash exactly the normalized rows consumed by :func:`run_diagnostic`."""
    return hashlib.sha256(
        json.dumps(
            {
                symbol: [[day, value] for day, value in sorted(values.items())]
                for symbol, values in sorted(normalized.items())
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _normalize_rv_series(
    rv_series: Mapping[str, Any],
) -> tuple[dict[str, dict[int, float]], dict[str, bool], dict[str, str]]:
    normalized: dict[str, dict[int, float]] = {}
    input_authority: dict[str, bool] = {}
    input_quality: dict[str, str] = {}
    if isinstance(rv_series, Mapping):
        for symbol, payload in rv_series.items():
            if type(symbol) is not str or not symbol:
                continue
            values, is_authoritative, quality = _normalize_symbol_series(payload)
            normalized[symbol] = values
            input_authority[symbol] = is_authoritative
            input_quality[symbol] = quality
    return normalized, input_authority, input_quality


def _mint_cli_authority(
    rv_series: Mapping[str, Any],
    *,
    source_authority: str,
    loader: str,
    loader_version: str,
    source_identity: str,
    source_content_sha256: str,
) -> _DiagnosticAuthority:
    """Mint a capability only after a CLI loader has fully validated its grid."""
    if (
        _ALLOWED_CLI_LOADERS.get(source_authority) != loader
        or loader_version != _CLI_LOADER_VERSION
        or type(source_identity) is not str
        or not source_identity
        or type(source_content_sha256) is not str
        or len(source_content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_content_sha256)
    ):
        raise ValueError("unallowlisted CLI source lineage")
    normalized, input_authority, _ = _normalize_rv_series(rv_series)
    members = set(DEFAULT_SPEC["session_calendar"]["members"])
    if set(normalized) != members or not all(input_authority.get(symbol) for symbol in members):
        raise ValueError("incomplete explicit-day source grid")
    day_sets = [set(normalized[symbol]) for symbol in sorted(members)]
    if not day_sets or not day_sets[0] or any(days != day_sets[0] for days in day_sets[1:]):
        raise ValueError("inconsistent source day grid")
    days = sorted(day_sets[0])
    if not all(current == previous + 1 for previous, current in pairwise(days)):
        raise ValueError("partial source day grid")
    lineage = {
        "authority": source_authority,
        "loader": loader,
        "loader_version": loader_version,
        "source_identity": source_identity,
        "source_content_sha256": source_content_sha256,
    }
    return _DiagnosticAuthority(_AUTHORITY_SEAL, lineage, _normalized_input_digest(normalized))


def _validated_authority(
    authority: object | None, input_digest: str
) -> tuple[dict[str, str], bool]:
    """Accept only a capability minted for these exact normalized inputs."""
    required = {
        "authority",
        "loader",
        "loader_version",
        "source_identity",
        "source_content_sha256",
    }
    if (
        not isinstance(authority, _DiagnosticAuthority)
        or authority._seal is not _AUTHORITY_SEAL
        or authority._input_digest != input_digest
        or set(authority._lineage) != required
        or _ALLOWED_CLI_LOADERS.get(authority._lineage["authority"]) != authority._lineage["loader"]
        or authority._lineage["loader_version"] != _CLI_LOADER_VERSION
        or any(
            type(authority._lineage[field]) is not str or not authority._lineage[field]
            for field in required
        )
    ):
        return {"authority": "unsealed_library_call"}, False
    digest = authority._lineage["source_content_sha256"]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        return {"authority": "unsealed_library_call"}, False
    return dict(authority._lineage), True


def run_diagnostic(
    rv_series: Mapping[str, Any],
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
    spec: Mapping[str, Any] | None = None,
    authority: object | None = None,
) -> VerdictReport:
    """Run the whole-search V-DIAG over every pre-registered pair.

    ``rv_series`` maps symbol -> daily RV input (see
    :func:`_normalize_symbol_series` for accepted shapes).  ``pairs``
    overrides the pre-registered SPEC pair list (each item ``(leader,
    follower)``); the SPEC (with any overrides merged) is echoed verbatim
    into the artifact.  Canonical admission requires an opaque capability
    minted by the verified CLI loader for these exact normalized inputs.
    Missing symbols, non-overlapping day ranges, or short series close each
    affected pair as ``insufficient_data``.  Never raises.
    """
    merged_spec, valid_spec, authoritative = _validated_spec(spec, pairs)
    normalized, input_authority, input_quality = _normalize_rv_series(rv_series)
    input_digest = _normalized_input_digest(normalized)
    normalized_source_lineage, sealed_source = _validated_authority(authority, input_digest)
    authoritative = authoritative and sealed_source
    merged_spec["canonical_admission_eligible"] = authoritative

    pair_items = [
        (str(item[0]), str(item[1]))
        for item in merged_spec["pairs"]
        if isinstance(item, (list, tuple)) and len(item) == 2
    ]
    results: list[PairDiagnostic] = []
    for leader, follower in pair_items:
        leader_map = normalized.get(leader, {})
        follower_map = normalized.get(follower, {})
        shared_days = sorted(set(leader_map).intersection(follower_map))
        artifact: dict[str, Any] | None = None
        contiguous = all(current == previous + 1 for previous, current in pairwise(shared_days))
        if valid_spec and shared_days and contiguous:
            follower_rv = [follower_map[day] for day in shared_days]
            leader_rv = [leader_map[day] for day in shared_days]
            artifact = evaluate_pair(
                follower_rv,
                leader_rv,
                spec={
                    key: value
                    for key, value in merged_spec.items()
                    if key != "canonical_admission_eligible"
                },
            )
        if artifact is None:
            results.append(
                PairDiagnostic(
                    leader=leader,
                    follower=follower,
                    status="insufficient_data",
                    n_shared_days=len(shared_days),
                    n_design_rows=0,
                    fold_qlike_improvements=(),
                    median_qlike_improvement=None,
                    median_mse_improvement=None,
                    fold_win_rate=None,
                    mean_qlike_baseline=None,
                    mean_qlike_candidate=None,
                    mean_qlike_ewma_reference=None,
                    bootstrap_pvalue=None,
                    bh_adjusted_pvalue=None,
                    shared_day_start=shared_days[0] if shared_days else None,
                    shared_day_end=shared_days[-1] if shared_days else None,
                )
            )
        else:
            results.append(
                PairDiagnostic(
                    leader=leader,
                    follower=follower,
                    status="evaluated",
                    n_shared_days=len(shared_days),
                    n_design_rows=int(artifact["n_design_rows"]),
                    fold_qlike_improvements=tuple(artifact["fold_qlike_improvements"]),
                    median_qlike_improvement=artifact["median_qlike_improvement"],
                    median_mse_improvement=artifact["median_mse_improvement"],
                    fold_win_rate=artifact["fold_win_rate"],
                    mean_qlike_baseline=artifact["mean_qlike_baseline"],
                    mean_qlike_candidate=artifact["mean_qlike_candidate"],
                    mean_qlike_ewma_reference=artifact["mean_qlike_ewma_reference"],
                    bootstrap_pvalue=artifact["bootstrap_pvalue"],
                    bh_adjusted_pvalue=None,
                    coefficient_sign_stability=_freeze(artifact["coefficient_sign_stability"]),
                    rv_excess_kurtosis=artifact["rv_excess_kurtosis"],
                    qlike_diff_skewness=artifact["qlike_diff_skewness"],
                    shared_day_start=shared_days[0],
                    shared_day_end=shared_days[-1],
                    fold_day_bounds=tuple(
                        (
                            shared_days[max(DEFAULT_HAR_LAGS) + fold["test_start"]],
                            shared_days[max(DEFAULT_HAR_LAGS) + fold["test_end"] - 1],
                        )
                        for fold in artifact["folds"]
                    ),
                )
            )

    # Whole-search BH-FDR uses the full registered family. Missing hypotheses
    # receive p=1 rather than shrinking multiplicity to the observed subset.
    evaluated_indices = [idx for idx, pair in enumerate(results) if pair.status == "evaluated"]
    family_pvalues = [
        float(pair.bootstrap_pvalue)
        if pair.status == "evaluated" and pair.bootstrap_pvalue is not None
        else 1.0
        for pair in results
    ]
    adjusted = bh_adjusted_pvalues(family_pvalues)
    complete_family = len(evaluated_indices) == len(results)
    improvement_floor = float(merged_spec["qlike_improvement_floor"])
    win_floor = float(merged_spec["fold_win_rate_floor"])
    bh_alpha = float(merged_spec["bh_alpha"])
    for idx in evaluated_indices:
        pair = results[idx]
        bh_p = float(adjusted[idx])
        admitted = (
            authoritative
            and complete_family
            and input_authority.get(pair.leader, False)
            and input_authority.get(pair.follower, False)
            and pair.median_qlike_improvement is not None
            and pair.median_qlike_improvement >= improvement_floor
            and pair.fold_win_rate is not None
            and pair.fold_win_rate >= win_floor
            and bh_p <= bh_alpha
        )
        results[idx] = PairDiagnostic(
            leader=pair.leader,
            follower=pair.follower,
            status=pair.status,
            n_shared_days=pair.n_shared_days,
            n_design_rows=pair.n_design_rows,
            fold_qlike_improvements=pair.fold_qlike_improvements,
            median_qlike_improvement=pair.median_qlike_improvement,
            median_mse_improvement=pair.median_mse_improvement,
            fold_win_rate=pair.fold_win_rate,
            mean_qlike_baseline=pair.mean_qlike_baseline,
            mean_qlike_candidate=pair.mean_qlike_candidate,
            mean_qlike_ewma_reference=pair.mean_qlike_ewma_reference,
            bootstrap_pvalue=pair.bootstrap_pvalue,
            bh_adjusted_pvalue=bh_p,
            coefficient_sign_stability=_freeze(pair.coefficient_sign_stability),
            rv_excess_kurtosis=pair.rv_excess_kurtosis,
            qlike_diff_skewness=pair.qlike_diff_skewness,
            admitted=bool(admitted),
            shared_day_start=pair.shared_day_start,
            shared_day_end=pair.shared_day_end,
            fold_day_bounds=pair.fold_day_bounds,
        )

    admitted_keys = tuple(f"{pair.leader}->{pair.follower}" for pair in results if pair.admitted)
    insufficient_keys = tuple(
        f"{pair.leader}->{pair.follower}" for pair in results if pair.status != "evaluated"
    )
    evaluated_count = len(evaluated_indices)
    if evaluated_count == 0 or not complete_family or not authoritative:
        verdict = "insufficient_data"
    elif admitted_keys:
        verdict = "admitted"
    else:
        verdict = "dead"
    gate_open = (
        authoritative
        and complete_family
        and any(pair.admitted and pair.leader == "BTCUSDT" for pair in results)
    )
    return VerdictReport(
        spec=_freeze(merged_spec),
        pairs=tuple(results),
        evaluated_pairs=evaluated_count,
        admitted_pairs=admitted_keys,
        insufficient_pairs=insufficient_keys,
        program_verdict=verdict,
        sizing_overlay_build_gate_open=bool(gate_open),
        lineage=_freeze(
            {
                "input_digest_sha256": input_digest,
                "loader_aggregation_version": _LINEAGE_VERSION,
                "source": normalized_source_lineage,
                "authority_counts": {
                    "explicit_day_series": sum(input_authority.values()),
                    "nonauthoritative_series": sum(
                        1 for value in input_authority.values() if not value
                    ),
                },
                "quality_counts": {
                    quality: sum(1 for item in input_quality.values() if item == quality)
                    for quality in sorted(set(input_quality.values()))
                },
                "per_symbol": {
                    symbol: {
                        "authority": (
                            "explicit_epoch_day"
                            if input_authority.get(symbol)
                            else "nonauthoritative"
                        ),
                        "quality": input_quality.get(symbol, "missing"),
                        "day_start": min(values) if values else None,
                        "day_end": max(values) if values else None,
                        "observations": len(values),
                    }
                    for symbol, values in sorted(normalized.items())
                },
            }
        ),
    )
