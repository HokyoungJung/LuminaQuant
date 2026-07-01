"""Look-ahead / causality purity guard for the Alpha101 factor library.

For each registered alpha we build a deterministic synthetic OHLCV panel and
verify that the factor value at time ``t`` depends only on data ``<= t``. We do
this with a *future-perturbation* test: two full-length panels that are byte
identical for rows ``[0..t]`` but independently random for rows ``[t+1..]`` must
produce an identical factor value at row ``t``. Any difference means the value
at ``t`` was contaminated by future data (a look-ahead leak).

Because every trailing rolling primitive emits row ``t`` after processing only
rows ``<= t``, causal alphas are bit-identical between the two panels. The sole
non-causal primitive in this library is ``indneutralize`` -- a *cross-sectional*
operation (neutralize against sector/industry across the universe at a fixed
timestamp). When exercised on a single-symbol time series its group-mean spans
the whole column, i.e. it reads future rows. That is an artifact of testing a
cross-sectional operator on the time axis, not a production temporal leak, so
the alphas that invoke it are allowlisted (see ``CROSS_SECTIONAL_ALPHAS``).

This module is strictly read-only: it never mutates factor source logic and
uses only public builders plus the code-native program layer to evaluate full
factor series. Covers AC-31 and AC-32.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lumina_quant.indicators import formulaic_definitions as fd
from lumina_quant.indicators.alpha101.compiler import build_context
from lumina_quant.indicators.alpha101.formula_sources import ALPHA_PROGRAM_DEFINITIONS
from lumina_quant.indicators.alpha101.registry import list_alpha_ids

# --- Deterministic synthetic panel configuration -------------------------------

_PANEL_ROWS = 340
_RANK_WINDOW = 20
_BASE_SEED = 42
_ALT_SEED = 20240517
# Target rows carry maximal history (least degenerate) yet leave a healthy block
# of future rows to perturb.
_TARGETS = (_PANEL_ROWS - 70, _PANEL_ROWS - 40, _PANEL_ROWS - 15)

# Relative/absolute tolerance for the causal-equality comparison. Causal alphas
# are bit-identical here; the tolerance only absorbs benign float noise.
_RTOL = 1e-9
_ATOL = 1e-11

# Alphas that invoke the cross-sectional ``indneutralize`` operator. On a single
# time series this operator is non-causal by construction (its group mean spans
# the whole column). Allowlisted for the causal check; validated against runtime
# detection by ``test_cross_sectional_allowlist_matches_indneutralize_usage``.
CROSS_SECTIONAL_ALPHAS = frozenset(
    {48, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100}
)

# Subset of the cross-sectional alphas whose look-ahead actually manifests as a
# numeric difference under future-perturbation (the operator output survives as
# a raw magnitude rather than through a shift-invariant op such as ts_corr/rank).
# Used as an adversarial sanity check that the harness truly detects leaks.
KNOWN_MANIFEST_LEAKERS = frozenset({48, 100})


def _make_raw(seed: int, n: int) -> dict[str, list[float]]:
    """Deterministic, well-conditioned synthetic OHLCV arrays."""

    rng = np.random.default_rng(seed)
    close = np.abs(100.0 + np.cumsum(rng.normal(0.0, 1.0, size=n))) + 10.0
    intraday = np.abs(rng.normal(0.0, 1.0, size=n)) + 0.5
    high = close + intraday
    low = np.minimum(close - intraday * 0.5, close - 0.01)
    opn = low + (high - low) * rng.uniform(0.2, 0.8, size=n)
    vol = 1_000_000.0 * (1.0 + np.abs(rng.normal(0.0, 0.3, size=n)))
    vwap = (high + low + close) / 3.0
    return {
        "opens": [float(v) for v in opn],
        "highs": [float(v) for v in high],
        "lows": [float(v) for v in low],
        "closes": [float(v) for v in close],
        "volumes": [float(v) for v in vol],
        "vwaps": [float(v) for v in vwap],
    }


def _splice_future(base_raw: dict[str, list[float]], alt_raw: dict[str, list[float]], t: int):
    """Panel identical to ``base_raw`` for rows [0..t], replaced by ``alt_raw`` after."""

    spliced: dict[str, list[float]] = {}
    for key, values in base_raw.items():
        merged = list(values)
        merged[t + 1 :] = alt_raw[key][t + 1 :]
        spliced[key] = merged
    return spliced


def _compute_full_series(alpha_id: int, context, *, track_indneutralize: bool = False):
    """Evaluate the code-native alpha program to a full pd.Series (all rows).

    The public spec callable collapses to the last finite scalar, so we reuse the
    code-native program layer (env + program) to obtain the whole factor series
    without touching any numeric source logic.
    """

    index = next(iter(context.values())).index
    env = fd._build_env(context, index=index, rank_window=_RANK_WINDOW)
    used = {"flag": False}
    if track_indneutralize:
        inner = env["indneutralize"]

        def _tracked(series, group):
            used["flag"] = True
            return inner(series, group)

        env["indneutralize"] = _tracked

    def const(key: str, default: float) -> float:
        return fd._resolve_constant(
            key=key, default=default, param_overrides=None, param_registry=None
        )

    definition = ALPHA_PROGRAM_DEFINITIONS[int(alpha_id)]
    result = definition.program(env, const)
    series = fd._to_series(result, index).replace([np.inf, -np.inf], np.nan)
    return series, used["flag"]


# --- Precomputed deterministic panels (shared across parametrized cases) --------

_BASE_RAW = _make_raw(_BASE_SEED, _PANEL_ROWS)
_ALT_RAW = _make_raw(_ALT_SEED, _PANEL_ROWS)
_BASE_CTX = build_context(**_BASE_RAW)
_PERTURBED_CTX = {t: build_context(**_splice_future(_BASE_RAW, _ALT_RAW, t)) for t in _TARGETS}


def _values_agree(a: float, b: float) -> bool:
    a_finite = math.isfinite(a)
    b_finite = math.isfinite(b)
    if not a_finite and not b_finite:
        return True
    if a_finite != b_finite:
        return False
    return bool(np.isclose(a, b, rtol=_RTOL, atol=_ATOL))


@pytest.mark.parametrize("alpha_id", list(list_alpha_ids()))
def test_alpha_value_is_causal(alpha_id: int):
    """Factor value at row t must be invariant to perturbing rows t+1..N-1."""

    if alpha_id in CROSS_SECTIONAL_ALPHAS:
        pytest.skip(
            "uses cross-sectional indneutralize (non-causal on a single-series time axis)"
        )

    base_series, _ = _compute_full_series(alpha_id, _BASE_CTX)

    usable_targets = [t for t in _TARGETS if math.isfinite(float(base_series.iloc[t]))]
    if not usable_targets:
        pytest.skip("degenerate: factor is NaN at every target row (insufficient warmup)")

    for t in usable_targets:
        perturbed_series, _ = _compute_full_series(alpha_id, _PERTURBED_CTX[t])
        base_value = float(base_series.iloc[t])
        perturbed_value = float(perturbed_series.iloc[t])
        assert _values_agree(base_value, perturbed_value), (
            f"alpha {alpha_id}: look-ahead leak at row {t} -- "
            f"base={base_value!r} perturbed={perturbed_value!r}"
        )


def test_lookahead_coverage_is_meaningful():
    """Guard against a vacuous suite: many causal alphas must actually be checked."""

    verified = 0
    for alpha_id in list_alpha_ids():
        if alpha_id in CROSS_SECTIONAL_ALPHAS:
            continue
        base_series, _ = _compute_full_series(alpha_id, _BASE_CTX)
        if any(math.isfinite(float(base_series.iloc[t])) for t in _TARGETS):
            verified += 1
    assert verified >= 60, f"only {verified} causal alphas were exercised"


def test_cross_sectional_allowlist_matches_indneutralize_usage():
    """The allowlist must equal the set of alphas that actually call indneutralize."""

    detected = set()
    for alpha_id in list_alpha_ids():
        _, used = _compute_full_series(alpha_id, _BASE_CTX, track_indneutralize=True)
        if used:
            detected.add(int(alpha_id))
    assert detected == set(CROSS_SECTIONAL_ALPHAS), (
        f"indneutralize usage {sorted(detected)} does not match allowlist "
        f"{sorted(CROSS_SECTIONAL_ALPHAS)}"
    )


def test_harness_detects_known_cross_sectional_leak():
    """Adversarial self-check: the perturbation harness must catch a real leak.

    Alphas 48 and 100 route the cross-sectional indneutralize output into a
    magnitude-sensitive position, so perturbing future rows changes the value at
    row ``t``. If this stops being detected the causal test has gone blind.
    """

    detected_a_leak = False
    for alpha_id in sorted(KNOWN_MANIFEST_LEAKERS):
        base_series, _ = _compute_full_series(alpha_id, _BASE_CTX)
        for t in _TARGETS:
            base_value = float(base_series.iloc[t])
            if not math.isfinite(base_value):
                continue
            perturbed_series, _ = _compute_full_series(alpha_id, _PERTURBED_CTX[t])
            perturbed_value = float(perturbed_series.iloc[t])
            if not _values_agree(base_value, perturbed_value):
                detected_a_leak = True
    assert detected_a_leak, (
        "perturbation harness failed to detect the known cross-sectional look-ahead"
    )
