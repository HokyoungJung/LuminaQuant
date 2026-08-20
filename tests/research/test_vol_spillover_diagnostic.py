"""Deterministic tests for the V-DIAG HAR-RV leader-spillover diagnostic.

Pre-registered falsification harness from the v-diag-runner proposal:
- synthetic NULL (two independent AR(1)-in-log-RV series) must produce ~zero
  admissions across seeds;
- PLANTED (follower log-RV explicitly loading on the leader's lag) must be
  admitted;
- determinism: the same inputs yield a byte-identical JSON artifact;
- degenerate inputs fail closed as ``insufficient_data`` (never raise);
- hand-computed QLIKE golden values and a BH-FDR golden vector;
- no-lookahead: mutating data after fold ``t`` leaves fold ``t`` forecasts
  bit-identical.

All randomness in the data generators is a seeded inline LCG (never the
``random`` module); the diagnostic's own bootstrap seed is fixed in its SPEC.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lumina_quant.research.vol_spillover_diagnostic import (
    DEFAULT_SPEC,
    bh_adjusted_pvalues,
    evaluate_pair,
    qlike_loss,
    run_diagnostic,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "research" / "run_vol_spillover_diagnostic.py"

# --------------------------------------------------------------------------- #
# LCG generators (deterministic, no `random` module)
# --------------------------------------------------------------------------- #


def _lcg_stream(seed: int):
    state = seed & 0x7FFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


def _gauss(stream) -> float:
    return sum(next(stream) for _ in range(12)) - 6.0


def _null_ar1_log_rv(n: int, seed: int, *, mu: float, phi: float, sigma: float) -> list[float]:
    """Independent AR(1)-in-log-RV series (the NULL generator)."""
    stream = _lcg_stream(seed)
    level = mu
    out: list[float] = []
    for _ in range(n):
        level = mu + phi * (level - mu) + sigma * _gauss(stream)
        out.append(math.exp(level))
    return out


def _planted_pair(n: int, seed_leader: int, seed_follower: int) -> tuple[list[float], list[float]]:
    """PLANTED generator: follower log-RV loads on the leader's LAG."""
    mu_l, mu_f = -9.0, -9.3
    stream_l = _lcg_stream(seed_leader)
    stream_f = _lcg_stream(seed_follower)
    level_l, level_f = mu_l, mu_f
    leader: list[float] = []
    follower: list[float] = []
    for _ in range(n):
        new_l = mu_l + 0.7 * (level_l - mu_l) + 0.5 * _gauss(stream_l)
        new_f = mu_f + 0.2 * (level_f - mu_f) + 0.8 * (level_l - mu_l) + 0.1 * _gauss(stream_f)
        leader.append(math.exp(new_l))
        follower.append(math.exp(new_f))
        level_l, level_f = new_l, new_f
    return leader, follower


# --------------------------------------------------------------------------- #
# NULL / PLANTED falsifiers
# --------------------------------------------------------------------------- #


def test_null_generator_yields_zero_admissions_across_seeds() -> None:
    for seed in (11, 22, 33):
        leader = _null_ar1_log_rv(700, 1000 + seed, mu=-9.0, phi=0.95, sigma=0.35)
        follower = _null_ar1_log_rv(700, 9000 + seed, mu=-9.3, phi=0.95, sigma=0.35)
        report = run_diagnostic(
            {"BTCUSDT": leader, "ETHUSDT": follower}, pairs=[("BTCUSDT", "ETHUSDT")]
        )
        assert report.program_verdict == "dead"
        assert report.admitted_pairs == ()
        assert report.sizing_overlay_build_gate_open is False
        pair = report.pairs[0]
        assert pair.status == "evaluated"
        assert pair.admitted is False
        assert pair.bh_adjusted_pvalue is not None


def test_planted_generator_is_admitted() -> None:
    leader, follower = _planted_pair(700, 42, 4242)
    report = run_diagnostic(
        {"BTCUSDT": leader, "ETHUSDT": follower}, pairs=[("BTCUSDT", "ETHUSDT")]
    )
    assert report.program_verdict == "admitted"
    assert report.admitted_pairs == ("BTCUSDT->ETHUSDT",)
    assert report.sizing_overlay_build_gate_open is True
    pair = report.pairs[0]
    assert pair.admitted is True
    assert pair.median_qlike_improvement is not None
    assert pair.median_qlike_improvement >= DEFAULT_SPEC["qlike_improvement_floor"]
    assert pair.fold_win_rate is not None
    assert pair.fold_win_rate >= DEFAULT_SPEC["fold_win_rate_floor"]
    assert pair.bh_adjusted_pvalue is not None
    assert pair.bh_adjusted_pvalue <= DEFAULT_SPEC["bh_alpha"]
    # Diagnostic tail columns are populated on evaluated pairs.
    assert pair.rv_excess_kurtosis is not None
    assert pair.qlike_diff_skewness is not None
    # SPEC is echoed verbatim (with the pair override) into the artifact.
    assert report.spec["bootstrap_seed"] == DEFAULT_SPEC["bootstrap_seed"]
    assert report.spec["pairs"] == [["BTCUSDT", "ETHUSDT"]]


def test_metals_pair_key_and_gate_scoping() -> None:
    # An admitted XAU pair must NOT open the BTC->alt sizing-overlay gate.
    leader, follower = _planted_pair(700, 7, 77)
    report = run_diagnostic(
        {"XAUUSDT": leader, "XAGUSDT": follower}, pairs=[("XAUUSDT", "XAGUSDT")]
    )
    assert report.pairs[0].status == "evaluated"
    assert report.sizing_overlay_build_gate_open is False


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def test_same_inputs_yield_byte_identical_json() -> None:
    leader, follower = _planted_pair(700, 42, 4242)
    first = run_diagnostic({"BTCUSDT": leader, "ETHUSDT": follower}, pairs=[("BTCUSDT", "ETHUSDT")])
    second = run_diagnostic(
        {"BTCUSDT": list(leader), "ETHUSDT": list(follower)}, pairs=[("BTCUSDT", "ETHUSDT")]
    )
    assert first.to_json() == second.to_json()


# --------------------------------------------------------------------------- #
# degenerate inputs fail closed (never raise)
# --------------------------------------------------------------------------- #


def test_degenerate_inputs_close_as_insufficient_data() -> None:
    # Missing symbols entirely.
    empty = run_diagnostic({}, pairs=[("BTCUSDT", "ETHUSDT")])
    assert empty.program_verdict == "insufficient_data"
    assert empty.pairs[0].status == "insufficient_data"
    # Short history.
    short = run_diagnostic(
        {"BTCUSDT": [1e-4] * 50, "ETHUSDT": [2e-4] * 50}, pairs=[("BTCUSDT", "ETHUSDT")]
    )
    assert short.pairs[0].status == "insufficient_data"
    # Non-overlapping day ranges via (days, rv) tuples.
    leader_days = (list(range(700)), [1e-4] * 700)
    follower_days = (list(range(10_000, 10_700)), [2e-4] * 700)
    disjoint = run_diagnostic(
        {"BTCUSDT": leader_days, "ETHUSDT": follower_days}, pairs=[("BTCUSDT", "ETHUSDT")]
    )
    assert disjoint.pairs[0].status == "insufficient_data"
    assert disjoint.pairs[0].n_shared_days == 0
    # NaN-poisoned RV values are dropped and the remainder is too short.
    nan_rv = [float("nan")] * 700
    poisoned = run_diagnostic(
        {"BTCUSDT": nan_rv, "ETHUSDT": [1e-4] * 700}, pairs=[("BTCUSDT", "ETHUSDT")]
    )
    assert poisoned.pairs[0].status == "insufficient_data"
    # Garbage payload shapes never raise.
    garbage = run_diagnostic(
        {"BTCUSDT": object(), "ETHUSDT": "nonsense"}, pairs=[("BTCUSDT", "ETHUSDT")]
    )
    assert garbage.pairs[0].status == "insufficient_data"
    # Constant RV series survive evaluation-or-closure without raising.
    constant = run_diagnostic(
        {"BTCUSDT": [1e-4] * 700, "ETHUSDT": [2e-4] * 700}, pairs=[("BTCUSDT", "ETHUSDT")]
    )
    assert constant.pairs[0].status in {"evaluated", "insufficient_data"}
    assert constant.pairs[0].admitted is False


def test_non_three_lag_specs_close_as_insufficient_data_never_raise() -> None:
    # REGRESSION (VDIAG-01): ``evaluate_pair`` must normalize ``har_lags``
    # exactly as ``har_design`` does (sorted unique ints) and fail closed
    # BEFORE building the design when the normalized count != 3.  Previously
    # a 2-lag spec raised IndexError inside ``_sign_stability`` (never-raise
    # breach) and a 4-lag spec silently mislabeled own-history columns as
    # leader coefficients.
    leader, follower = _planted_pair(700, 42, 4242)
    # Too few lags: [1, 5] -> insufficient_data, no raise.
    assert evaluate_pair(follower, leader, spec={"har_lags": [1, 5]}) is None
    short_report = run_diagnostic(
        {"BTCUSDT": leader, "ETHUSDT": follower},
        pairs=[("BTCUSDT", "ETHUSDT")],
        spec={"har_lags": [1, 5]},
    )
    assert short_report.pairs[0].status == "insufficient_data"
    assert short_report.program_verdict == "insufficient_data"
    # Too many lags: [1, 5, 22, 66] -> insufficient_data (no mislabeled
    # coefficient table is ever produced).
    assert evaluate_pair(follower, leader, spec={"har_lags": [1, 5, 22, 66]}) is None
    long_report = run_diagnostic(
        {"BTCUSDT": leader, "ETHUSDT": follower},
        pairs=[("BTCUSDT", "ETHUSDT")],
        spec={"har_lags": [1, 5, 22, 66]},
    )
    assert long_report.pairs[0].status == "insufficient_data"
    # Non-coercible lag entries also fail closed instead of raising.
    assert evaluate_pair(follower, leader, spec={"har_lags": [1, "junk", 22]}) is None


def test_default_lag_artifact_unchanged_by_normalization() -> None:
    # The default (1, 5, 22) path must stay byte-identical, and a spec that
    # normalizes TO the default (unsorted with a duplicate: sorted unique of
    # [22, 5, 1, 5] is [1, 5, 22]) must produce the exact same numbers.
    leader, follower = _planted_pair(700, 42, 4242)
    default_artifact = evaluate_pair(follower, leader)
    normalized_artifact = evaluate_pair(follower, leader, spec={"har_lags": [22, 5, 1, 5]})
    assert default_artifact is not None and normalized_artifact is not None
    for key in (
        "n_design_rows",
        "fold_qlike_improvements",
        "median_qlike_improvement",
        "median_mse_improvement",
        "fold_win_rate",
        "mean_qlike_baseline",
        "mean_qlike_candidate",
        "bootstrap_pvalue",
        "coefficient_sign_stability",
    ):
        assert default_artifact[key] == normalized_artifact[key], key
    fold0_default = default_artifact["folds"][0]
    fold0_normalized = normalized_artifact["folds"][0]
    assert np.array_equal(
        fold0_default["forecast_candidate"], fold0_normalized["forecast_candidate"]
    )
    # Whole-artifact byte identity on the explicit default triple.
    report_default = run_diagnostic(
        {"BTCUSDT": leader, "ETHUSDT": follower}, pairs=[("BTCUSDT", "ETHUSDT")]
    )
    report_explicit = run_diagnostic(
        {"BTCUSDT": leader, "ETHUSDT": follower},
        pairs=[("BTCUSDT", "ETHUSDT")],
        spec={"har_lags": [1, 5, 22]},
    )
    assert report_default.to_json() == report_explicit.to_json()


def test_day_keyed_mapping_input_is_supported() -> None:
    leader, follower = _planted_pair(700, 42, 4242)
    leader_map = {day: value for day, value in enumerate(leader)}
    follower_map = {day: value for day, value in enumerate(follower)}
    report = run_diagnostic(
        {"BTCUSDT": leader_map, "ETHUSDT": follower_map}, pairs=[("BTCUSDT", "ETHUSDT")]
    )
    assert report.pairs[0].status == "evaluated"
    assert report.pairs[0].admitted is True


# --------------------------------------------------------------------------- #
# QLIKE goldens (hand-computed) + hand-rolled reference parity
# --------------------------------------------------------------------------- #


def test_qlike_hand_goldens() -> None:
    loss = qlike_loss([2.0], [1.0])
    assert loss is not None
    assert loss[0] == pytest.approx(1.0 - math.log(2.0), abs=1e-15)
    exact = qlike_loss([1.0], [1.0])
    assert exact is not None and exact[0] == 0.0
    # Patton QLIKE is minimized at h == sigma2, asymmetric around it.
    under = qlike_loss([1.0], [0.5])
    over = qlike_loss([1.0], [2.0])
    assert under is not None and over is not None
    assert under[0] > 0.0 and over[0] > 0.0
    assert under[0] != over[0]


def test_qlike_matches_hand_rolled_reference_to_1e12() -> None:
    stream = _lcg_stream(2026)
    realized = [1e-4 * (0.5 + next(stream)) for _ in range(64)]
    forecast = [1e-4 * (0.5 + next(stream)) for _ in range(64)]
    loss = qlike_loss(realized, forecast)
    assert loss is not None
    for sigma2, h, got in zip(realized, forecast, loss, strict=True):
        reference = sigma2 / h - math.log(sigma2 / h) - 1.0
        assert abs(got - reference) < 1e-12


def test_qlike_guards_fail_closed() -> None:
    assert qlike_loss([1.0, 2.0], [1.0]) is None  # shape mismatch
    assert qlike_loss([float("nan")], [1.0]) is None
    assert qlike_loss("junk", [1.0]) is None


# --------------------------------------------------------------------------- #
# BH-FDR golden vector
# --------------------------------------------------------------------------- #


def test_bh_fdr_golden_vector() -> None:
    adjusted = bh_adjusted_pvalues([0.01, 0.04, 0.03, 0.20])
    expected = [0.04, 0.16 / 3.0, 0.16 / 3.0, 0.20]
    assert adjusted == pytest.approx(expected, abs=1e-12)
    assert bh_adjusted_pvalues([]) == []
    assert bh_adjusted_pvalues([0.9, 0.95]) == pytest.approx([0.95, 0.95], abs=1e-12)
    assert bh_adjusted_pvalues([1.0]) == [1.0]


# --------------------------------------------------------------------------- #
# no-lookahead: future mutation leaves earlier folds bit-identical
# --------------------------------------------------------------------------- #


def test_mutating_future_folds_leaves_fold0_forecasts_unchanged() -> None:
    leader, follower = _planted_pair(700, 42, 4242)
    baseline = evaluate_pair(follower, leader)
    assert baseline is not None
    # Fold 0 only touches design rows < its test_end, i.e. series indices
    # <= test_end + max_lag - 1.  Mutating from index 300 (>+ margin) onward
    # must leave every fold-0 artifact bit-identical.
    fold0 = baseline["folds"][0]
    assert fold0["test_end"] + max(DEFAULT_SPEC["har_lags"]) < 300
    mutated_follower = list(follower)
    mutated_leader = list(leader)
    for idx in range(300, 700):
        mutated_follower[idx] *= 7.0
        mutated_leader[idx] *= 3.0
    mutated = evaluate_pair(mutated_follower, mutated_leader)
    assert mutated is not None
    fold0_mutated = mutated["folds"][0]
    assert np.array_equal(fold0["forecast_baseline"], fold0_mutated["forecast_baseline"])
    assert np.array_equal(fold0["forecast_candidate"], fold0_mutated["forecast_candidate"])
    assert np.array_equal(fold0["qlike_baseline"], fold0_mutated["qlike_baseline"])
    assert np.array_equal(fold0["qlike_candidate"], fold0_mutated["qlike_candidate"])


# --------------------------------------------------------------------------- #
# CLI runner: byte-identical artifact + insufficient-pair accounting
# --------------------------------------------------------------------------- #


def test_cli_runner_writes_byte_identical_artifact(tmp_path: Path) -> None:
    leader, follower = _planted_pair(700, 42, 4242)
    csv_path = tmp_path / "rv.csv"
    lines = ["symbol,day,rv"]
    lines.extend(f"BTCUSDT,{day},{value!r}" for day, value in enumerate(leader))
    lines.extend(f"ETHUSDT,{day},{value!r}" for day, value in enumerate(follower))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_first = tmp_path / "artifact_a.json"
    out_second = tmp_path / "artifact_b.json"
    for out_path in (out_first, out_second):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--rv-csv", str(csv_path), "--out", str(out_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    first_bytes = out_first.read_bytes()
    assert first_bytes == out_second.read_bytes()

    payload = json.loads(first_bytes)
    # Default pre-registered pairs: BTC->ETH evaluated, all others (including
    # every metals pair) close as insufficient_data on this two-symbol input.
    assert payload["program_verdict"] == "admitted"
    assert payload["admitted_pairs"] == ["BTCUSDT->ETHUSDT"]
    assert payload["sizing_overlay_build_gate_open"] is True
    assert len(payload["pairs"]) == len(DEFAULT_SPEC["pairs"])
    insufficient = set(payload["insufficient_pairs"])
    assert "XAUUSDT->XAGUSDT" in insufficient
    assert len(insufficient) == len(DEFAULT_SPEC["pairs"]) - 1
    # SPEC echoed verbatim.
    assert payload["spec"]["bootstrap_seed"] == DEFAULT_SPEC["bootstrap_seed"]
    assert payload["spec"]["qlike_improvement_floor"] == 0.05
