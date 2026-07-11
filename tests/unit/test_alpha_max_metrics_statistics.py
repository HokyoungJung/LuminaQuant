from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lumina_quant.portfolio import optimizer_core
from lumina_quant.research import alpha_max_evidence as evidence
from lumina_quant.research.alpha_max_evidence import (
    ALPHA_MAX_DSR_NUM_TRIALS,
    ALPHA_MAX_PERIODS_PER_YEAR,
    AlphaMaxEquityEndpoint,
    AlphaMaxPrimaryReturnStream,
    AlphaMaxStreamingEquityEvidence,
    AlphaMaxStreamingEquityTracker,
    AlphaMaxTrialLedger,
    alpha_max_common_rng_seed,
    alpha_max_common_rng_seed_payload,
    alpha_max_drawdown_duration,
    alpha_max_full_event_mdd,
    alpha_max_pre_gate_sharpe_variance,
    alpha_max_trial_key,
    alpha_max_trial_key_set_lf_bytes,
    alpha_max_type7_quantile,
    build_alpha_max_primary_return_stream,
    build_alpha_max_statistical_evidence,
    build_alpha_max_trial_ledger,
    compute_alpha_max_capacity_diagnostics,
    compute_alpha_max_metric_statistics,
    compute_alpha_max_turnover_rpt,
    normalize_alpha_max_prior_trial_node,
    read_alpha_max_prior_trial_blob,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/research/alpha_max_portfolio_20260710.json"
PRIOR_FILE_SHA256 = "f2c86ae7bb9f9719143fa0b11e73c68ad021160aeac03a0aa5c6fa93636d57b6"
PRIOR_KEY_SET_SHA256 = "3b078011040f89e8d788b2cef9214c58f687221104381e26a688a7f8cdbddd78"
CURRENT_REGISTRY_SHA256 = "cfe3a04620c52cc235d6f1cda1cac617ba30cd7327c753fc2f620d8250d51a4e"
CURRENT_KEY_SET_SHA256 = "3a4791cf353abcb82f9717ce89ee16b9d73d84f431d5b058135046c2ba8e332b"


def _calendar(count: int, *, start: datetime | None = None) -> tuple[datetime, ...]:
    first = start or datetime(2025, 6, 8, tzinfo=UTC)
    return tuple(first + timedelta(hours=4 * index) for index in range(count))


def _stream(
    returns: list[float] | tuple[float, ...],
    *,
    start: datetime | None = None,
) -> AlphaMaxPrimaryReturnStream:
    equity = 10_000.0
    endpoints: list[AlphaMaxEquityEndpoint] = []
    calendar = _calendar(len(returns), start=start)
    for timestamp, value in zip(calendar, returns, strict=True):
        equity *= 1.0 + value
        endpoints.append(AlphaMaxEquityEndpoint(timestamp=timestamp, equity=equity))
    return build_alpha_max_primary_return_stream(endpoints, calendar)


def _streaming_equity(values: list[float] | tuple[float, ...]) -> AlphaMaxStreamingEquityEvidence:
    tracker = AlphaMaxStreamingEquityTracker()
    for timestamp_ms, value in enumerate(values, start=1):
        tracker.observe((timestamp_ms, value))
    return tracker.finalize()


@pytest.fixture(scope="module")
def frozen_inputs() -> tuple[bytes, dict[str, Any], AlphaMaxTrialLedger]:
    blob = read_alpha_max_prior_trial_blob(REPO_ROOT)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return blob, config, build_alpha_max_trial_ledger(blob, config)


def test_u42_common_rng_payload_hash_modulo_and_row_independence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = alpha_max_common_rng_seed_payload("validation_w01", 30)
    assert payload == b"alpha_max_20260710\x00validation_w01\x0030"
    assert hashlib.sha256(payload).hexdigest() == (
        "ed90033525088e003b23c1b1a265d44b5f39193a4d13a85216cbb0a9f204bb40"
    )
    assert alpha_max_common_rng_seed("validation_w01", 30) == 2_659_438
    assert alpha_max_common_rng_seed("train", 30) == 1_937_660_375
    assert alpha_max_common_rng_seed("validation_w01", 20) == 1_578_309_719
    assert alpha_max_common_rng_seed("validation_w01", 30) == alpha_max_common_rng_seed(
        "validation_w01", 30
    )

    class _ZeroDigest:
        @staticmethod
        def digest() -> bytes:
            return bytes(32)

    monkeypatch.setattr(evidence.hashlib, "sha256", lambda _: _ZeroDigest())
    assert alpha_max_common_rng_seed("validation_w01", 30) == 1


@pytest.mark.parametrize(
    ("split", "cost"),
    [
        ("", 30),
        (" validation_w01", 30),
        ("validation_w01\x00row", 30),
        ("validation_w01", 0),
        ("validation_w01", 25),
        ("validation_w01", True),
    ],
)
def test_u42_common_rng_rejects_noncanonical_inputs(split: Any, cost: Any) -> None:
    with pytest.raises(ValueError, match="alpha_max_rng_"):
        alpha_max_common_rng_seed_payload(split, cost)


def test_u43_complete_utc_four_hour_primary_stream_is_arithmetic_from_flat_capital() -> None:
    calendar = _calendar(4)
    equities = (10_100.0, 9_999.0, 10_098.99, 10_098.99)
    stream = build_alpha_max_primary_return_stream(
        [
            AlphaMaxEquityEndpoint(timestamp=timestamp, equity=equity)
            for timestamp, equity in zip(calendar, equities, strict=True)
        ],
        calendar,
    )
    expected = (
        10_100.0 / 10_000.0 - 1.0,
        9_999.0 / 10_100.0 - 1.0,
        10_098.99 / 9_999.0 - 1.0,
        0.0,
    )
    assert stream.endpoint_timestamps == calendar
    assert stream.endpoint_equities == equities
    assert stream.returns == expected
    assert stream.initial_capital == 10_000.0
    assert stream.periods_per_year == 2190 == ALPHA_MAX_PERIODS_PER_YEAR
    assert set(stream.to_payload()) == {
        "artifact_kind",
        "calendar_sha256",
        "endpoint_equities",
        "endpoint_timestamps",
        "initial_capital",
        "periods_per_year",
        "returns",
    }


def test_u43_missing_duplicate_daily_truncated_or_substituted_calendar_rejects() -> None:
    calendar = _calendar(3)
    endpoints = [AlphaMaxEquityEndpoint(timestamp=value, equity=10_000.0) for value in calendar]
    with pytest.raises(ValueError, match="endpoint_count_mismatch"):
        build_alpha_max_primary_return_stream(endpoints[:-1], calendar)
    with pytest.raises(ValueError, match="calendar_not_strict"):
        build_alpha_max_primary_return_stream(endpoints, (calendar[0], calendar[0], calendar[2]))
    with pytest.raises(ValueError, match="calendar_incomplete"):
        build_alpha_max_primary_return_stream(
            endpoints,
            (calendar[0], calendar[1], calendar[1] + timedelta(days=1)),
        )
    with pytest.raises(ValueError, match="calendar_mismatch"):
        build_alpha_max_primary_return_stream(endpoints[1:], calendar[:-1])
    with pytest.raises(ValueError, match="not_4h_endpoint"):
        build_alpha_max_primary_return_stream(
            [replace(endpoints[0], timestamp=calendar[0] + timedelta(hours=1)), *endpoints[1:]],
            calendar,
        )


@pytest.mark.parametrize("equity", [0.0, -1.0, math.inf, math.nan, True])
def test_u43_nonpositive_or_nonfinite_endpoint_rejects(equity: Any) -> None:
    calendar = _calendar(1)
    with pytest.raises(ValueError, match="primary_endpoint_equity_invalid"):
        build_alpha_max_primary_return_stream(
            [AlphaMaxEquityEndpoint(timestamp=calendar[0], equity=equity)],
            calendar,
        )


def test_u43_non_utc_and_wrong_initial_capital_reject() -> None:
    non_utc = datetime(2025, 6, 8, tzinfo=timezone(timedelta(hours=9)))
    with pytest.raises(ValueError, match="not_utc"):
        build_alpha_max_primary_return_stream(
            [AlphaMaxEquityEndpoint(timestamp=non_utc, equity=10_000.0)],
            [non_utc],
        )
    calendar = _calendar(1)
    with pytest.raises(ValueError, match="initial_capital_mismatch"):
        build_alpha_max_primary_return_stream(
            [AlphaMaxEquityEndpoint(timestamp=calendar[0], equity=10_000.0)],
            calendar,
            initial_capital=9_999.0,
        )


def test_u44_metrics_wrapper_calls_only_canonical_optimizer_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _stream([0.02, -0.01, 0.015, -0.02, 0.01, 0.005] * 3)
    original = optimizer_core.metrics
    calls: list[tuple[np.ndarray, int]] = []

    def _spy(values: np.ndarray, *, periods_per_year: int) -> dict[str, float]:
        calls.append((values.copy(), periods_per_year))
        return original(values, periods_per_year=periods_per_year)

    monkeypatch.setattr(evidence.optimizer_core, "metrics", _spy)
    result = compute_alpha_max_metric_statistics(
        stream,
        _streaming_equity([10_500.0, 9_000.0, stream.endpoint_equities[-1]]),
    )
    expected = original(np.asarray(stream.returns), periods_per_year=2190)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], np.asarray(stream.returns))
    assert calls[0][1] == 2190
    assert dict(result.canonical_metrics) == expected
    assert result.reporting_4h_mdd == expected["max_drawdown"]
    assert result.full_event_mdd == pytest.approx(1.0 - 9_000.0 / 10_500.0)
    assert result.gate_mdd == max(result.full_event_mdd, result.reporting_4h_mdd)


def test_u44_poisoned_primary_stream_and_nonfinite_canonical_output_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _stream([0.01, -0.01] * 8)
    original = optimizer_core.metrics
    poisoned = replace(stream, returns=(*stream.returns[:-1], math.nan))
    with pytest.raises(ValueError, match="return_identity_mismatch"):
        compute_alpha_max_metric_statistics(
            poisoned,
            _streaming_equity(stream.endpoint_equities),
        )

    def _bad(_: np.ndarray, *, periods_per_year: int) -> dict[str, float]:
        assert periods_per_year == 2190
        output = original(np.asarray(stream.returns), periods_per_year=2190)
        output["sharpe"] = math.nan
        return output

    monkeypatch.setattr(evidence.optimizer_core, "metrics", _bad)
    with pytest.raises(ValueError, match="canonical_metric_sharpe_invalid"):
        compute_alpha_max_metric_statistics(
            stream,
            _streaming_equity(stream.endpoint_equities),
        )


def test_u46_drawdown_duration_type7_var_and_worst_ceiling_es() -> None:
    equities = [11_000.0, 10_000.0, 9_000.0, 9_500.0, 11_000.0, 10_000.0, 9_000.0]
    calendar = _calendar(len(equities))
    stream = build_alpha_max_primary_return_stream(
        [AlphaMaxEquityEndpoint(timestamp=t, equity=v) for t, v in zip(calendar, equities)],
        calendar,
    )
    assert alpha_max_drawdown_duration(stream) == (3, 12)
    assert alpha_max_type7_quantile([0.0, 10.0, 20.0, 30.0], 0.25) == 7.5
    assert alpha_max_full_event_mdd([10_500.0, 8_400.0, 9_000.0]) == pytest.approx(0.20)

    tail_stream = _stream([float(index) / 1_000.0 for index in range(-10, 10)])
    result = compute_alpha_max_metric_statistics(
        tail_stream,
        _streaming_equity(tail_stream.endpoint_equities),
    )
    assert result.value_at_risk_5pct_type7 == pytest.approx(-0.00905)
    assert result.expected_shortfall_5pct == pytest.approx(-0.01)
    assert result.drawdown_duration_hours == 4 * result.drawdown_duration_endpoints


def test_u47_turnover_and_rpt_are_report_only_and_zero_is_exactly_undefined() -> None:
    records = [
        {"applied_qty": 2.0, "fill_price": 100.0},
        {"applied_qty": 3.0, "fill_price": 50.0},
        {"applied_qty": 0.0, "fill_price": 1.0},
    ]
    result = compute_alpha_max_turnover_rpt(
        records,
        initial_capital=10_000.0,
        ending_equity=10_350.0,
    )
    expected_turnover = math.fsum(abs(row["applied_qty"] * row["fill_price"]) for row in records)
    assert result.turnover_notional == expected_turnover == 350.0
    assert result.turnover_multiple == 0.035
    assert result.rpt_bps == 10_000.0
    assert result.undefined_reason is None
    assert not ({"eligible", "gate", "rank", "deployment"} & set(result.to_payload()))

    zero = compute_alpha_max_turnover_rpt([], initial_capital=10_000.0, ending_equity=9_000.0)
    assert zero.rpt_bps is None
    assert zero.undefined_reason == "undefined_zero_turnover"
    assert zero.turnover_notional == zero.turnover_multiple == 0.0


@pytest.mark.parametrize(
    "record",
    [
        {"applied_qty": 1.0},
        {"applied_qty": 1.0, "fill_price": 2.0, "rank": 1},
        {"applied_qty": -1.0, "fill_price": 2.0},
        {"applied_qty": 1.0, "fill_price": math.inf},
    ],
)
def test_u47_turnover_schema_is_fail_closed(record: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="alpha_max_turnover_"):
        compute_alpha_max_turnover_rpt(
            [record],
            initial_capital=10_000.0,
            ending_equity=10_000.0,
        )


def test_u48_capacity_formula_type7_summary_and_report_only_role() -> None:
    orders = [
        {
            "bar_volume": 1_000.0,
            "raw_price": 20.0,
            "equity_before": 10_000.0,
            "requested_qty": 100.0,
        },
        {
            "bar_volume": 2_000.0,
            "raw_price": 10.0,
            "equity_before": 12_000.0,
            "requested_qty": 100.0,
        },
        {"bar_volume": 3_000.0, "raw_price": 5.0, "equity_before": 9_000.0, "requested_qty": 300.0},
        {"bar_volume": 0.0, "raw_price": 1.0, "equity_before": 10_000.0, "requested_qty": 0.0},
    ]
    result = compute_alpha_max_capacity_diagnostics(orders)
    assert result.observation_count == 3
    assert dict(result.capacity_proxy_equity_usdt or {}) == {
        "minimum": 9_000.0,
        "p10_type7": 9_200.0,
        "median_type7": 10_000.0,
    }
    assert result.undefined_reason is None
    assert not ({"eligible", "gate", "rank", "deployment"} & set(result.to_payload()))

    empty = compute_alpha_max_capacity_diagnostics([])
    assert empty.capacity_proxy_equity_usdt is None
    assert empty.observation_count == 0
    assert empty.undefined_reason == "undefined_no_positive_order"


@pytest.mark.parametrize(
    "record",
    [
        {"bar_volume": 1.0, "raw_price": 1.0, "equity_before": 1.0},
        {"bar_volume": 0.0, "raw_price": 1.0, "equity_before": 1.0, "requested_qty": 1.0},
        {"bar_volume": 1.0, "raw_price": math.nan, "equity_before": 1.0, "requested_qty": 1.0},
        {"bar_volume": 1.0, "raw_price": 1.0, "equity_before": 1.0, "requested_qty": -1.0},
    ],
)
def test_u48_capacity_schema_and_positive_observations_are_fail_closed(
    record: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="alpha_max_capacity_"):
        compute_alpha_max_capacity_diagnostics([record])


def test_u17_u18_immutable_blob_actual_lf_and_frozen_counts_hashes(
    frozen_inputs: tuple[bytes, dict[str, Any], AlphaMaxTrialLedger],
) -> None:
    blob, config, ledger = frozen_inputs
    assert hashlib.sha256(blob).hexdigest() == PRIOR_FILE_SHA256
    assert config["current_trial_registry"]["nodes"] == json.loads(
        json.dumps(config["current_trial_registry"]["nodes"])
    )
    assert len(ledger.prior_trial_keys) == 1466
    assert len(ledger.current_trial_keys) == 21
    assert len(ledger.union_trial_keys) == ledger.num_trials == ALPHA_MAX_DSR_NUM_TRIALS == 1487
    assert not set(ledger.prior_trial_keys).intersection(ledger.current_trial_keys)
    assert ledger.prior_key_set_sha256 == PRIOR_KEY_SET_SHA256
    assert ledger.current_key_set_sha256 == CURRENT_KEY_SET_SHA256
    assert ledger.current_registry_sha256 == CURRENT_REGISTRY_SHA256
    assert ledger.prior_key_set_lf_bytes.endswith(b"\x0a")
    assert ledger.current_key_set_lf_bytes.endswith(b"\x0a")
    assert b"\\n" not in ledger.prior_key_set_lf_bytes
    assert b"\\n" not in ledger.current_key_set_lf_bytes
    assert ledger.prior_key_set_lf_bytes.count(b"\x0a") == 1466
    assert ledger.current_key_set_lf_bytes.count(b"\x0a") == 21
    assert alpha_max_trial_key_set_lf_bytes(ledger.prior_trial_keys) == (
        ledger.prior_key_set_lf_bytes
    )


def test_u17_blob_reader_ignores_worktree_and_ambient_path_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "read_bytes", lambda _: b"ambient-poison")
    blob = read_alpha_max_prior_trial_blob(REPO_ROOT)
    assert hashlib.sha256(blob).hexdigest() == PRIOR_FILE_SHA256


def test_u18_prior_normalization_excludes_cosmetic_fields_but_preserves_behavior(
    frozen_inputs: tuple[bytes, dict[str, Any], AlphaMaxTrialLedger],
) -> None:
    blob, _, _ = frozen_inputs
    candidate = json.loads(blob)["candidates"][0]
    node = normalize_alpha_max_prior_trial_node(candidate)
    assert set(node) == {
        "schema",
        "kind",
        "implementation",
        "timeframe",
        "symbols",
        "params",
        "behavior_metadata",
        "members",
        "allocation",
        "gross",
        "omission",
    }
    cosmetic = copy.deepcopy(candidate)
    cosmetic.pop("name")
    cosmetic["status"] = "poisoned-cosmetic-status"
    cosmetic["tags"] = ["renamed"]
    assert alpha_max_trial_key(normalize_alpha_max_prior_trial_node(cosmetic)) == (
        alpha_max_trial_key(node)
    )
    behavioral = copy.deepcopy(candidate)
    behavioral["metadata"]["alpha_max_test_mutation"] = True
    assert alpha_max_trial_key(normalize_alpha_max_prior_trial_node(behavioral)) != (
        alpha_max_trial_key(node)
    )
    cross_kind = dict(node)
    cross_kind["kind"] = "current_matrix_row"
    assert alpha_max_trial_key(cross_kind) != alpha_max_trial_key(node)


def test_u17_u18_mutated_blob_registry_or_binding_fails_closed(
    frozen_inputs: tuple[bytes, dict[str, Any], AlphaMaxTrialLedger],
) -> None:
    blob, config, ledger = frozen_inputs
    mutated_blob = bytearray(blob)
    mutated_blob[-2] ^= 1
    with pytest.raises(ValueError, match="prior_trial_inventory_mismatch"):
        build_alpha_max_trial_ledger(bytes(mutated_blob), config)

    mutated_config = copy.deepcopy(config)
    mutated_config["current_trial_registry"]["nodes"][0]["params"]["atr_period"] += 1
    with pytest.raises(ValueError, match="current_trial_inventory_mismatch"):
        build_alpha_max_trial_ledger(blob, mutated_config)

    streams = {"candidate": _stream([0.01, -0.005] * 8)}
    forged = replace(ledger, prior_key_set_lf_bytes=b"poison\n")
    with pytest.raises(ValueError, match="trial_ledger_binding_invalid"):
        build_alpha_max_statistical_evidence(streams, forged)


def test_u45_pre_gate_variance_and_canonical_statistical_roles(
    monkeypatch: pytest.MonkeyPatch,
    frozen_inputs: tuple[bytes, dict[str, Any], AlphaMaxTrialLedger],
) -> None:
    ledger = frozen_inputs[2]
    streams = {
        "zeta": _stream([0.010, -0.004, 0.006, -0.002] * 8),
        "alpha": _stream([0.008, -0.003, 0.004, -0.001] * 8),
        "middle": _stream([0.012, -0.006, 0.003, -0.002] * 8),
    }
    sharpe_evidence = alpha_max_pre_gate_sharpe_variance(streams)
    sharpes = [
        float(np.mean(streams[key].returns)) / float(np.std(streams[key].returns, ddof=1))
        for key in sorted(streams)
    ]
    assert sharpe_evidence.variance_across_trials == pytest.approx(np.var(sharpes, ddof=1))
    assert not sharpe_evidence.degenerate_candidate_ids

    dsr_calls: list[tuple[np.ndarray, dict[str, Any]]] = []
    spa_calls: list[tuple[np.ndarray, dict[str, Any]]] = []
    pbo_calls: list[tuple[np.ndarray, int]] = []

    def _dsr(values: np.ndarray, **kwargs: Any) -> float:
        dsr_calls.append((values.copy(), kwargs))
        return 0.91

    def _spa(values: np.ndarray, **kwargs: Any) -> float:
        spa_calls.append((values.copy(), kwargs))
        return 0.04

    def _pbo(matrix: np.ndarray, *, n_splits: int) -> float:
        pbo_calls.append((matrix.copy(), n_splits))
        return 0.20

    monkeypatch.setattr(evidence.research_metrics, "deflated_sharpe_ratio", _dsr)
    monkeypatch.setattr(evidence.research_metrics, "spa_like_pvalue", _spa)
    monkeypatch.setattr(evidence.research_metrics, "cscv_pbo", _pbo)
    result = build_alpha_max_statistical_evidence(streams, ledger)

    assert result.candidate_ids == ("alpha", "middle", "zeta")
    assert result.input_role == "pre_gate_matched_selection_eligible"
    assert result.nominal_cost_bps == 30
    assert result.dsr_num_trials == 1487
    assert result.dsr_hac_inference is True
    assert result.spa_bootstrap_rounds == 2000
    assert result.spa_block_size == max(1, round(32 ** (1.0 / 3.0)))
    assert result.spa_seed == 12345
    assert result.pbo_n_splits == 8
    assert result.prior_trial_key_set_sha256 == PRIOR_KEY_SET_SHA256
    assert result.current_trial_key_set_sha256 == CURRENT_KEY_SET_SHA256
    assert len(dsr_calls) == len(spa_calls) == 3
    for _, kwargs in dsr_calls:
        assert kwargs == {
            "num_trials": 1487,
            "variance_across_trials": sharpe_evidence.variance_across_trials,
            "hac_inference": True,
        }
    for _, kwargs in spa_calls:
        assert kwargs == {"bootstrap_rounds": 2000, "block_size": 3, "seed": 12345}
    assert len(pbo_calls) == 1 and pbo_calls[0][1] == 8
    np.testing.assert_array_equal(
        pbo_calls[0][0],
        np.vstack([np.asarray(streams[key].returns) for key in sorted(streams)]),
    )
    assert not (
        {"correlation_discount", "effective_trials", "rank", "selected"} & set(result.to_payload())
    )


def test_u45_calendar_mismatch_degenerate_and_nonfinite_outputs_fail(
    monkeypatch: pytest.MonkeyPatch,
    frozen_inputs: tuple[bytes, dict[str, Any], AlphaMaxTrialLedger],
) -> None:
    ledger = frozen_inputs[2]
    mismatched = {
        "alpha": _stream([0.01, -0.005] * 8),
        "beta": _stream(
            [0.01, -0.005] * 8,
            start=datetime(2025, 6, 8, 4, tzinfo=UTC),
        ),
    }
    with pytest.raises(ValueError, match="statistical_calendar_mismatch"):
        build_alpha_max_statistical_evidence(mismatched, ledger)

    degenerate = {"flat": _stream([0.0] * 16)}
    pre_gate = alpha_max_pre_gate_sharpe_variance(degenerate)
    assert pre_gate.variance_across_trials == 0.0
    assert pre_gate.degenerate_candidate_ids == ("flat",)
    with pytest.raises(ValueError, match="statistical_stream_degenerate"):
        build_alpha_max_statistical_evidence(degenerate, ledger)

    valid = {"alpha": _stream([0.01, -0.005] * 8)}
    monkeypatch.setattr(
        evidence.research_metrics, "deflated_sharpe_ratio", lambda *_a, **_k: math.nan
    )
    with pytest.raises(ValueError, match="dsr_output_invalid"):
        build_alpha_max_statistical_evidence(valid, ledger)
