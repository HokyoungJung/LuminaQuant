"""HAC/Newey-West rank-IC t-stat correction (research.hac_inference).

Covers both seams the strategy-viability review flagged as treating overlapping
horizon-``h`` forward-return ICs as iid:

* ``lumina_quant.research.factor_ic._summarize`` (descriptive ``t_stat`` output),
* ``lumina_quant.alpha_zoo.evidence.summarize_split_evidence`` (the ``t_stat``
  that drives the alive/reversed selection gate).

Two contracts are proven for each:

1.  **Flag OFF is byte-identical to the legacy iid t-stat.** The correction is
    opt-in; with ``hac_inference=False`` (the default) every field, including
    ``t_stat``, is exactly the pre-change value, and the whole reducer payload is
    unchanged. A lag-0 request is also an exact identity.
2.  **Flag ON widens the t-stat for autocorrelated ICs.** A Bartlett-kernel
    Newey-West standard error (lag ``label_horizon``) deflates the naive t by
    ``sqrt(VIF)``; only ``t_stat`` moves (IC mean/std/IR/positive-ratio/turnover/
    spread/decay are untouched), and a Monte-Carlo reproduces the review's
    headline: iid over-rejects the null at h=4 while HAC restores it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lumina_quant.alpha_zoo.evidence import (
    AlphaEvidenceThresholds,
    _classify,
    _hac_variance_inflation as _hac_vif_evidence,
    summarize_split_evidence,
)
from lumina_quant.research.factor_ic import (
    _hac_variance_inflation as _hac_vif_factor,
    _reduce_factor_ic_batched,
    _reduce_factor_ic_loop,
    _summarize,
    evaluate_factor_ic_numpy,
)


def _legacy_iid_t(arr: np.ndarray) -> float:
    n = int(arr.shape[0])
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    return float(mean / (std / math.sqrt(float(n))))


def _ar1(mu: float, phi: float, n: int, seed: int, noise: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.float64)
    x[0] = mu
    for i in range(1, n):
        x[i] = mu + phi * (x[i - 1] - mu) + rng.standard_normal() * noise
    return x


def _synthetic_panel(
    *, n_symbols: int = 12, n_timestamps: int = 60, n_factors: int = 3, seed: int = 20260706
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    symbols = np.array([f"SYM{i:02d}" for i in range(n_symbols)])
    timestamps = np.arange(n_timestamps, dtype=np.int64) * 60_000
    sym_col = np.repeat(symbols, n_timestamps)
    ts_col = np.tile(timestamps, n_symbols)
    n_rows = n_symbols * n_timestamps
    factors = rng.standard_normal((n_rows, n_factors))
    fwd = 0.3 * factors[:, 0] + 0.15 * factors[:, 1] + rng.standard_normal(n_rows) * 0.9
    factors[3, 0] = np.inf
    factors[7, 1] = np.nan
    fwd[11] = -np.inf
    names = [f"f{i}" for i in range(n_factors)]
    return {
        "symbol": sym_col,
        "timestamp": ts_col,
        "factors": factors,
        "forward_return": fwd,
        "factor_names": names,
    }


# ---------------------------------------------------------------------------
# The variance-inflation kernel (identical in both modules).
# ---------------------------------------------------------------------------


def test_hac_vif_lag_zero_is_exact_identity() -> None:
    arr = np.random.default_rng(1).standard_normal(50)
    mean = float(np.mean(arr))
    for vif in (_hac_vif_factor(arr, mean, 0), _hac_vif_evidence(arr, mean, 0)):
        assert vif == 1.0


def test_hac_vif_matches_manual_bartlett_newey_west() -> None:
    # Small fixed series; VIF = 1 + 2 * sum_k (1 - k/(L+1)) * gamma_k / gamma_0.
    arr = np.array([0.10, 0.14, 0.09, 0.16, 0.12, 0.18, 0.11, 0.15], dtype=np.float64)
    mean = float(np.mean(arr))
    lag = 3
    centered = arr - mean
    s0 = float(np.sum(centered * centered))
    manual = 1.0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = float(np.sum(centered[k:] * centered[:-k]))
        manual += 2.0 * w * (gamma_k / s0)
    assert _hac_vif_factor(arr, mean, lag) == manual
    assert _hac_vif_evidence(arr, mean, lag) == manual


def test_hac_vif_is_one_when_series_has_no_dispersion() -> None:
    arr = np.full(10, 0.07, dtype=np.float64)
    assert _hac_vif_factor(arr, 0.07, 4) == 1.0
    assert _hac_vif_evidence(arr, 0.07, 4) == 1.0


# ---------------------------------------------------------------------------
# factor_ic._summarize
# ---------------------------------------------------------------------------


def test_summarize_flag_off_is_byte_identical_to_legacy() -> None:
    arr = np.random.default_rng(2).standard_normal(40) * 0.1 + 0.05
    default = _summarize(arr)
    explicit_off = _summarize(arr, hac_inference=False, label_horizon=4)
    assert default == explicit_off
    # t_stat (index 4) is exactly the legacy iid formula.
    assert default[4] == _legacy_iid_t(arr)


def test_summarize_lag_zero_leaves_tstat_unchanged() -> None:
    arr = _ar1(0.05, 0.7, 40, seed=3)
    off = _summarize(arr, hac_inference=False)
    on_lag0 = _summarize(arr, hac_inference=True, label_horizon=0)
    assert on_lag0 == off


def test_summarize_hac_deflates_tstat_by_sqrt_vif() -> None:
    arr = _ar1(0.05, 0.8, 200, seed=4)
    off = _summarize(arr, hac_inference=False)
    on = _summarize(arr, hac_inference=True, label_horizon=4)
    # Descriptive stats (mean, std, ir, positive_ratio) are untouched.
    assert on[0] == off[0]
    assert on[1] == off[1]
    assert on[2] == off[2]
    assert on[3] == off[3]
    # t_stat is deflated: |t_hac| < |t_iid| for a positively autocorrelated series.
    assert abs(on[4]) < abs(off[4])
    vif = _hac_vif_factor(arr, float(np.mean(arr)), 4)
    assert vif > 1.0
    assert on[4] == off[4] / math.sqrt(vif)


# ---------------------------------------------------------------------------
# factor_ic reducer end-to-end (byte identity + parity of the ON path)
# ---------------------------------------------------------------------------


def test_reducer_flag_off_payload_byte_identical() -> None:
    panel = _synthetic_panel()
    default = evaluate_factor_ic_numpy(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
    ).to_dict()
    explicit_off = evaluate_factor_ic_numpy(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
        hac_inference=False,
        label_horizon=4,
    ).to_dict()
    assert default == explicit_off


def test_reducer_hac_only_changes_tstat() -> None:
    panel = _synthetic_panel()
    off = evaluate_factor_ic_numpy(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
    ).as_mapping()
    on = evaluate_factor_ic_numpy(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
        hac_inference=True,
        label_horizon=4,
    ).as_mapping()
    assert set(off) == set(on)
    any_tstat_changed = False
    for name in off:
        a, b = off[name], on[name]
        assert a.ic_mean == b.ic_mean, name
        assert a.ic_std == b.ic_std, name
        assert a.ic_ir == b.ic_ir, name
        assert a.ic_positive_ratio == b.ic_positive_ratio, name
        assert a.turnover_mean == b.turnover_mean, name
        assert a.quantile_spread_mean == b.quantile_spread_mean, name
        assert a.rank_autocorr == b.rank_autocorr, name
        if a.t_stat != b.t_stat:
            any_tstat_changed = True
    assert any_tstat_changed


def test_hac_on_path_is_loop_batched_bit_identical() -> None:
    # The parity oracle (loop) and production path (batched) must agree bit-for-bit
    # with the correction engaged, exactly as they do with it off.
    panel = _synthetic_panel(n_symbols=8, n_timestamps=120, n_factors=3, seed=99)
    kwargs = dict(hac_inference=True, label_horizon=4)
    loop = _reduce_factor_ic_loop(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
        **kwargs,
    )
    batched = _reduce_factor_ic_batched(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
        **kwargs,
    )
    assert loop.to_dict() == batched.to_dict()


# ---------------------------------------------------------------------------
# evidence.summarize_split_evidence + the alive/reversed gate
# ---------------------------------------------------------------------------


def test_summarize_split_flag_off_byte_identical() -> None:
    frame = pd.DataFrame({"rank_ic": _ar1(0.05, 0.7, 40, seed=5), "n": 8})
    default = summarize_split_evidence(frame).to_dict()
    explicit_off = summarize_split_evidence(frame, hac_inference=False, label_horizon=4).to_dict()
    assert default == explicit_off
    values = frame["rank_ic"].to_numpy(dtype=float)
    assert default["t_stat"] == _legacy_iid_t(values)


def test_summarize_split_hac_deflates_only_tstat() -> None:
    frame = pd.DataFrame({"rank_ic": _ar1(0.05, 0.8, 200, seed=6)})
    off = summarize_split_evidence(frame)
    on = summarize_split_evidence(frame, hac_inference=True, label_horizon=4)
    assert on.ic_mean == off.ic_mean
    assert on.ic_std == off.ic_std
    assert on.ic_ir == off.ic_ir
    assert on.ic_positive_ratio == off.ic_positive_ratio
    assert abs(on.t_stat) < abs(off.t_stat)


def test_hac_flips_over_admitted_alive_candidate_to_dead() -> None:
    # A positively-autocorrelated null-ish IC series whose iid t clears the 2.0
    # hurdle purely because overlap understates its SE; HAC correctly rejects it.
    frame = pd.DataFrame({"rank_ic": _ar1(0.028, 0.7, 40, seed=9)})
    thresholds = AlphaEvidenceThresholds(
        min_periods=8, min_abs_ic_mean=0.0, min_positive_ratio=0.0, min_abs_t_stat=2.0
    )
    s_iid = summarize_split_evidence(frame)
    s_hac = summarize_split_evidence(frame, hac_inference=True, label_horizon=4)

    assert s_iid.t_stat > 2.0
    assert s_hac.t_stat < 2.0

    cls_iid, passed_iid, _ = _classify(s_iid, thresholds)
    cls_hac, passed_hac, _ = _classify(s_hac, thresholds)
    assert (cls_iid, passed_iid) == ("alive", True)
    assert (cls_hac, passed_hac) == ("dead", False)


# ---------------------------------------------------------------------------
# Monte-Carlo: iid over-rejects the null at horizon 4; HAC restores it.
# ---------------------------------------------------------------------------


def test_hac_restores_null_false_positive_rate_at_horizon_4() -> None:
    rng = np.random.default_rng(20260706)
    horizon = 4
    n_periods = 120
    trials = 6000
    iid_fp = 0
    hac_fp = 0
    for _ in range(trials):
        shocks = rng.standard_normal(n_periods + horizon - 1)
        # Overlapping length-``horizon`` window sums -> MA(horizon-1) autocorrelation,
        # population mean 0 (the null: no real IC).
        ic = np.array([shocks[i : i + horizon].sum() for i in range(n_periods)])
        if abs(_legacy_iid_t(ic)) > 1.96:
            iid_fp += 1
        _, _, _, _, t_hac = _summarize(ic, hac_inference=True, label_horizon=horizon)
        if abs(t_hac) > 1.96:
            hac_fp += 1
    iid_rate = iid_fp / trials
    hac_rate = hac_fp / trials
    # iid grossly over-rejects (nominal 0.05); HAC pulls it back toward nominal.
    assert iid_rate > 0.25
    assert hac_rate < 0.18
    assert iid_rate - hac_rate > 0.12
