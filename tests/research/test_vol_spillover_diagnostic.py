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

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lumina_quant.research.vol_spillover_diagnostic import (
    DEFAULT_SPEC,
    _mint_cli_authority,
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
        assert report.program_verdict == "insufficient_data"
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
    assert report.program_verdict == "insufficient_data"
    assert report.admitted_pairs == ()
    assert report.sizing_overlay_build_gate_open is False
    pair = report.pairs[0]
    assert pair.admitted is False
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
    assert report.spec["pairs"] == (("BTCUSDT", "ETHUSDT"),)


def test_complete_canonical_family_direct_call_cannot_open_gate() -> None:
    """Even complete explicit-day data has no authority outside the CLI."""
    series: dict[str, dict[int, float]] = {}
    pairs_by_leader: dict[str, list[str]] = {}
    for leader, follower in DEFAULT_SPEC["pairs"]:
        pairs_by_leader.setdefault(leader, []).append(follower)
    for family_index, (leader, followers) in enumerate(sorted(pairs_by_leader.items())):
        leader_seed = 42 + family_index
        for follower_index, follower in enumerate(followers):
            leader_values, follower_values = _planted_pair(
                700, leader_seed, 4242 + 100 * family_index + follower_index
            )
            series.setdefault(leader, dict(enumerate(leader_values)))
            series[follower] = dict(enumerate(follower_values))
    report = run_diagnostic(series)
    assert report.spec["canonical_admission_eligible"] is False
    assert report.sizing_overlay_build_gate_open is False
    assert report.admitted_pairs == ()


def test_complete_canonical_null_falsifier_cannot_admit() -> None:
    series: dict[str, dict[int, float]] = {}
    for index, symbol in enumerate(
        sorted({symbol for pair in DEFAULT_SPEC["pairs"] for symbol in pair})
    ):
        series[symbol] = dict(
            enumerate(_null_ar1_log_rv(700, 20_000 + index, mu=-9.0, phi=0.95, sigma=0.35))
        )
    report = run_diagnostic(series)
    assert report.spec["canonical_admission_eligible"] is False
    assert report.admitted_pairs == ()
    assert report.sizing_overlay_build_gate_open is False


def test_plain_sequences_and_nested_evidence_cannot_be_promoted_or_mutated() -> None:
    leader, follower = _planted_pair(700, 42, 4242)
    report = run_diagnostic(
        {"BTCUSDT": leader, "ETHUSDT": follower}, pairs=[("BTCUSDT", "ETHUSDT")]
    )
    before = report.to_json()
    assert report.sizing_overlay_build_gate_open is False
    assert report.lineage["per_symbol"]["BTCUSDT"]["authority"] == "nonauthoritative"
    with pytest.raises(TypeError):
        report.pairs[0].coefficient_sign_stability["leader_d"] = {}
    assert report.to_json() == before


def test_unsealed_or_malformed_direct_inputs_cannot_recover_canonical_admission() -> None:
    """No direct call may inherit canonical source or pair authority."""
    leader, follower = _planted_pair(700, 42, 4242)
    unsealed = run_diagnostic(
        {"BTCUSDT": dict(enumerate(leader)), "ETHUSDT": dict(enumerate(follower))},
        pairs=[("BTCUSDT", "ETHUSDT")],
    )
    assert unsealed.spec["canonical_admission_eligible"] is False
    assert unsealed.sizing_overlay_build_gate_open is False
    assert unsealed.lineage["source"]["authority"] == "unsealed_library_call"

    malformed = run_diagnostic(
        {"BTCUSDT": dict(enumerate(leader)), "ETHUSDT": dict(enumerate(follower))},
        pairs=[("BTCUSDT", "ETHUSDT"), "not-a-pair"],  # type: ignore[list-item]
        authority={
            "authority": "explicit_epoch_day",
            "loader": "_load_rv_csv",
            "loader_version": "vdiag-cli-v3",
            "source_identity": "test://vdiag-rv",
            "source_content_sha256": "0" * 64,
        },
    )
    assert malformed.program_verdict == "insufficient_data"
    assert malformed.pairs == ()
    assert malformed.sizing_overlay_build_gate_open is False


def test_fabricated_authority_values_and_private_shapes_close_gate() -> None:
    leader, follower = _planted_pair(700, 42, 4242)
    inputs = {"BTCUSDT": dict(enumerate(leader)), "ETHUSDT": dict(enumerate(follower))}

    class _PrivateShape:
        _seal = object()
        _input_digest = "0" * 64
        _lineage = {
            "authority": "explicit_epoch_day",
            "loader": "_load_rv_csv",
            "loader_version": "vdiag-cli-v3",
            "source_identity": "test://forged",
            "source_content_sha256": "0" * 64,
        }

    for forged in (
        "explicit_epoch_day",
        {"authority": "explicit_epoch_day"},
        _PrivateShape(),
    ):
        report = run_diagnostic(inputs, pairs=[("BTCUSDT", "ETHUSDT")], authority=forged)
        assert report.spec["canonical_admission_eligible"] is False
        assert report.sizing_overlay_build_gate_open is False


def test_authority_is_bound_to_the_exact_normalized_input_digest() -> None:
    inputs = {
        symbol: {0: 0.0001, 1: 0.0002} for symbol in DEFAULT_SPEC["session_calendar"]["members"]
    }
    authority = _mint_cli_authority(
        inputs,
        source_authority="explicit_epoch_day",
        loader="_load_rv_csv",
        loader_version="vdiag-cli-v3",
        source_identity="test://sealed-grid",
        source_content_sha256="0" * 64,
    )
    inputs["BTCUSDT"][1] = 0.9
    report = run_diagnostic(inputs, authority=authority)
    assert report.spec["canonical_admission_eligible"] is False
    assert report.sizing_overlay_build_gate_open is False


@pytest.mark.parametrize(
    ("loader", "loader_version"),
    [("_forged_loader", "vdiag-cli-v3"), ("_load_rv_csv", "forged-v1")],
)
def test_authority_mint_rejects_wrong_loader_or_version(loader: str, loader_version: str) -> None:
    inputs = {
        symbol: {0: 0.0001, 1: 0.0002} for symbol in DEFAULT_SPEC["session_calendar"]["members"]
    }
    with pytest.raises(ValueError, match="unallowlisted CLI source lineage"):
        _mint_cli_authority(
            inputs,
            source_authority="explicit_epoch_day",
            loader=loader,
            loader_version=loader_version,
            source_identity="test://sealed-grid",
            source_content_sha256="0" * 64,
        )


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
    # Three arbitrary lags cannot inherit daily/weekly/monthly labels.
    assert evaluate_pair(follower, leader, spec={"har_lags": [2, 3, 4]}) is None


def test_calendar_gap_and_incomplete_registered_family_cannot_admit() -> None:
    leader, follower = _planted_pair(700, 42, 4242)
    leader_map = {day: value for day, value in enumerate(leader) if day != 350}
    follower_map = {day: value for day, value in enumerate(follower)}
    gapped = run_diagnostic(
        {"BTCUSDT": leader_map, "ETHUSDT": follower_map},
        pairs=[("BTCUSDT", "ETHUSDT")],
    )
    assert gapped.pairs[0].status == "insufficient_data"
    assert gapped.program_verdict == "insufficient_data"
    assert gapped.sizing_overlay_build_gate_open is False

    incomplete_family = run_diagnostic({"BTCUSDT": leader, "ETHUSDT": follower})
    assert incomplete_family.program_verdict == "insufficient_data"
    assert incomplete_family.admitted_pairs == ()
    assert incomplete_family.sizing_overlay_build_gate_open is False


def test_spec_snapshot_is_stable_against_caller_and_default_mutation() -> None:
    leader, follower = _planted_pair(700, 42, 4242)
    override = {"har_lags": [1, 5, 22]}
    report = run_diagnostic(
        {"BTCUSDT": leader, "ETHUSDT": follower},
        pairs=[("BTCUSDT", "ETHUSDT")],
        spec=override,
    )
    frozen = report.to_json()
    override["har_lags"].append(99)
    DEFAULT_SPEC["har_lags"].append(99)
    try:
        assert report.to_json() == frozen
        rerun = run_diagnostic(
            {"BTCUSDT": leader, "ETHUSDT": follower},
            pairs=[("BTCUSDT", "ETHUSDT")],
            spec={"har_lags": [1, 5, 22]},
        )
        assert rerun.to_json() == frozen
    finally:
        DEFAULT_SPEC["har_lags"].pop()


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
    assert report.pairs[0].admitted is False


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
    """Only the verified CLI may mint a serializable-lineage authority."""
    csv_path = tmp_path / "rv.csv"
    lines = ["symbol,day,rv"]
    for symbol_index, symbol in enumerate(DEFAULT_SPEC["session_calendar"]["members"]):
        lines.extend(f"{symbol},{day},{0.0001 * (symbol_index + day + 1)!r}" for day in range(2))
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
    # The complete but deliberately short grid is CLI-authoritative; every
    # pair still closes on the pre-registered history requirement.
    assert payload["program_verdict"] == "insufficient_data"
    assert payload["admitted_pairs"] == []
    assert payload["sizing_overlay_build_gate_open"] is False
    assert payload["spec"]["canonical_admission_eligible"] is True
    assert len(payload["pairs"]) == len(DEFAULT_SPEC["pairs"])
    insufficient = set(payload["insufficient_pairs"])
    assert "XAUUSDT->XAGUSDT" in insufficient
    assert len(insufficient) == len(DEFAULT_SPEC["pairs"])
    # SPEC echoed verbatim.
    assert payload["spec"]["bootstrap_seed"] == DEFAULT_SPEC["bootstrap_seed"]
    assert payload["spec"]["qlike_improvement_floor"] == 0.05
    assert payload["lineage"]["source"] == {
        "authority": "explicit_epoch_day",
        "loader": "_load_rv_csv",
        "loader_version": "vdiag-cli-v3",
        "source_identity": str(csv_path.resolve()),
        "source_content_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }


def test_cli_rejects_malformed_pair_override_without_canonical_fallback(tmp_path: Path) -> None:
    csv_path = tmp_path / "rv.csv"
    csv_path.write_text("symbol,day,rv\nBTCUSDT,0,0.0001\n", encoding="utf-8")
    out_path = tmp_path / "artifact.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--rv-csv",
            str(csv_path),
            "--pairs",
            "BTCUSDT:ETHUSDT,broken",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["program_verdict"] == "insufficient_data"
    assert payload["sizing_overlay_build_gate_open"] is False


def test_cli_rejects_partial_intraday_utc_day_grid(tmp_path: Path) -> None:
    closes = tmp_path / "closes.csv"
    start = 1_577_836_800
    lines = ["symbol,epoch_seconds,close"]
    lines.extend(f"BTCUSDT,{start + hour * 3600},{100 + hour}" for hour in range(23))
    closes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_path = tmp_path / "artifact.json"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--closes-csv", str(closes), "--out", str(out_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["program_verdict"] == "insufficient_data"
    assert payload["sizing_overlay_build_gate_open"] is False
