from __future__ import annotations

import hashlib
import json
import math
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import lumina_quant.research.alpha_max_evidence as evidence
from lumina_quant.core.engine import TradingEngine
from lumina_quant.research.alpha_max_evidence import (
    ALPHA_MAX_CANDIDATE_SYMBOLS,
    AlphaMaxAdmissionCandidateInput,
    AlphaMaxCapsuleReceipt,
    AlphaMaxCostCellEvidence,
    AlphaMaxEquityEndpoint,
    AlphaMaxGateInput,
    AlphaMaxManifestMaterialization,
    AlphaMaxManifestReceipt,
    AlphaMaxOrderedFundingLookup,
    AlphaMaxRawObservation,
    AlphaMaxRootReceipt,
    AlphaMaxRowEvidence,
    AlphaMaxStreamingEquityTracker,
    FeatureRootSpec,
    alpha_max_common_rng_seed,
    alpha_max_full_event_mdd,
    alpha_max_terminal_outcome,
    build_alpha_max_prelock_seal,
    build_alpha_max_primary_return_stream,
    build_alpha_max_terminal_state,
    canonical_alpha_max_cost_cell_bytes,
    canonical_alpha_max_row_bytes,
    compute_alpha_max_metric_statistics,
    compute_alpha_max_train_admission,
    rank_alpha_max_historical_report,
    reconcile_alpha_max_cost_attribution,
    seal_alpha_max_contract_manifest,
    seal_alpha_max_root_tree,
    select_alpha_max_prelock_champion,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64


class _FeatureStrategy:
    required_inputs = ()
    required_features = ("funding_rate",)


def _feature_spec(tmp_path: Path, root_id: str) -> FeatureRootSpec:
    path = tmp_path / root_id
    path.mkdir()
    start, end = evidence._ROOT_INTERVALS[root_id]
    return FeatureRootSpec(root_id, str(path), "binance", start, end, _HASH_A, _HASH_B)


def test_b2_ordered_lookup_exposes_immutable_current_root_capability_to_real_engine_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Lookup:
        def __init__(self, *, db_path, exchange, start_date, end_date):
            self.db_path = db_path
            self.exchange = exchange

    monkeypatch.setattr(evidence, "FeaturePointLookup", _Lookup)
    warmup = _feature_spec(tmp_path, "warmup")
    train = _feature_spec(tmp_path, "train")
    lookup = AlphaMaxOrderedFundingLookup((warmup, train))

    assert lookup.db_path == train.path
    engine = object.__new__(TradingEngine)
    engine.strategy = _FeatureStrategy()
    TradingEngine._assert_strategy_requirements(
        engine,
        available_inputs=set(),
        feature_lookup=lookup,
    )
    with pytest.raises(AttributeError, match="immutable"):
        lookup.db_path = warmup.path


def test_root_tree_seal_is_canonical_streaming_and_rejects_unsafe_entries(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    (root / "BTCUSDT").mkdir(parents=True)
    (root / "ETHUSDT").mkdir()
    (root / "BTCUSDT" / "b.parquet").write_bytes(b"b")
    (root / "ETHUSDT" / "a.parquet").write_bytes(b"a")

    first = seal_alpha_max_root_tree("train", "raw", root)
    second = seal_alpha_max_root_tree("train", "raw", root)
    assert first == second
    assert [entry.relative_path for entry in first.entries] == [
        "BTCUSDT/b.parquet",
        "ETHUSDT/a.parquet",
    ]
    assert first.inventory_sha256 == second.inventory_sha256
    assert first.content_sha256 == second.content_sha256
    assert first.canonical_bytes == second.canonical_bytes
    assert first.to_receipt().content_sha256 == first.content_sha256

    (root / "BTCUSDT" / "b.parquet").write_bytes(b"mutated")
    mutated = seal_alpha_max_root_tree("train", "raw", root)
    assert mutated.content_sha256 != first.content_sha256
    assert mutated.inventory_sha256 != first.inventory_sha256

    (root / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        seal_alpha_max_root_tree("train", "raw", root)
    with pytest.raises(ValueError, match="must_be_absolute"):
        seal_alpha_max_root_tree("train", "raw", Path("relative"))


def _contract_manifest() -> dict[str, object]:
    return {
        "schema_version": "alpha_max_contract_manifest.v1",
        "exchange": "binance",
        "records": [
            {
                "symbol": symbol,
                "market_type": "perpetual",
                "linear": True,
                "inverse": False,
                "quote_asset": "USDT",
                "margin_asset": "USDT",
                "settle_asset": "USDT",
                "volume_unit": "base_asset",
                "contract_multiplier": 1.0,
            }
            for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS
        ],
    }


def test_contract_manifest_seal_accepts_only_exact_canonical_ten_symbol_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "contracts.json"
    canonical = (
        json.dumps(
            _contract_manifest(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    path.write_bytes(canonical)
    seal = seal_alpha_max_contract_manifest(path)
    assert seal.sha256 == hashlib.sha256(canonical).hexdigest()
    assert tuple(record.symbol for record in seal.records) == ALPHA_MAX_CANDIDATE_SYMBOLS
    with pytest.raises(FrozenInstanceError):
        seal.sha256 = _HASH_A

    poisoned = _contract_manifest()
    poisoned["records"][0]["volume_unit"] = "quote_asset"  # type: ignore[index]
    path.write_bytes(json.dumps(poisoned, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    with pytest.raises(ValueError, match="contract_manifest_mismatch"):
        seal_alpha_max_contract_manifest(path)


def _candidate_input(symbol: str, *, passes: bool, missing_last_bucket: bool = False):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    volume = 40_000.0 if passes else 20_000.0
    rows: list[AlphaMaxRawObservation] = []
    for day_index in range(517):
        for hour in (0, 4, 8, 12, 16, 20):
            if missing_last_bucket and day_index == 516 and hour == 20:
                continue
            rows.append(
                AlphaMaxRawObservation(
                    timestamp=start + timedelta(days=day_index, hours=hour),
                    close=100.0,
                    volume=volume,
                )
            )
    return AlphaMaxAdmissionCandidateInput(
        symbol=symbol,
        train_observations=tuple(rows),
        consecutive_completed_daily_bars_before_train=366,
        causal_funding_coverage_complete=True,
        unresolved_daily_cross_section_count=0,
    )


def test_actual_517_day_train_admission_computes_float64_fsum_type7_and_vectors() -> None:
    inputs = {
        symbol: _candidate_input(symbol, passes=index < 5)
        for index, symbol in enumerate(ALPHA_MAX_CANDIDATE_SYMBOLS)
    }
    result = compute_alpha_max_train_admission(
        inputs,
        input_root_hashes={"warmup": _HASH_A, "train": _HASH_B},
    )
    assert result.artifact.admitted_symbols == ALPHA_MAX_CANDIDATE_SYMBOLS[:5]
    assert len(result.daily_quote_notional_by_symbol["ADAUSDT"]) == 517
    assert result.daily_quote_notional_by_symbol["ADAUSDT"][0].quote_notional_usdt == (24_000_000.0)
    payload = json.loads(result.artifact.canonical_bytes)
    assert payload["per_candidate"]["ADAUSDT"]["statistics"] == {
        "causal_funding_coverage_complete": True,
        "complete_train_4h_keys": True,
        "complete_train_daily_keys": True,
        "consecutive_completed_daily_bars_before_train": 366,
        "daily_quote_notional_day_count": 517,
        "median_quote_notional_usdt": 24_000_000.0,
        "p10_quote_notional_usdt": 24_000_000.0,
        "readable_monotone_unique_finite_partitions": True,
        "unresolved_daily_cross_section_count": 0,
    }
    assert result.sha256 == hashlib.sha256(result.canonical_bytes).hexdigest()


def test_train_admission_missing_bucket_is_not_synthetic_zero_and_fails_membership() -> None:
    inputs = {
        symbol: _candidate_input(
            symbol,
            passes=index < 5,
            missing_last_bucket=index == 0,
        )
        for index, symbol in enumerate(ALPHA_MAX_CANDIDATE_SYMBOLS)
    }
    with pytest.raises(ValueError, match="insufficient_train_universe"):
        compute_alpha_max_train_admission(
            inputs,
            input_root_hashes={"warmup": _HASH_A, "train": _HASH_B},
        )


def _gate(
    row_id: str,
    *,
    role: str = "prelock_selection",
    total: float = 0.20,
    cagr: float = 0.15,
    calmar: float = 0.50,
    sharpe: float = 1.0,
    full_mdd: float = 0.20,
    report_mdd: float = 0.20,
    dsr: float = 0.95,
) -> AlphaMaxGateInput:
    return AlphaMaxGateInput(
        row_id=row_id,
        comparison_role=role,
        evidence_tier="actual_engine",
        comparison_valid=True,
        nominal_cost_bps=30,
        cumulative_return=total,
        cagr=cagr,
        calmar=calmar,
        net_sharpe=sharpe,
        full_event_mdd=full_mdd,
        reporting_4h_mdd=report_mdd,
        dsr=dsr,
        spa_pvalue=0.01,
        pbo=0.10,
        native_data_coverage_complete=True,
        funding_coverage_complete=True,
        hash_valid=True,
        manifest_valid=True,
        reconciliation_complete=True,
        ruin=False,
        raw_root_set_sha256=_HASH_A,
        feature_root_set_sha256=_HASH_B,
        universe_sha256="c" * 64,
        calendar_sha256="d" * 64,
        seed_schedule_sha256="e" * 64,
    )


def test_gate_order_soft_mdd_comparator_and_return_first_selection_are_exact() -> None:
    normal = _gate("normal", total=0.25, cagr=0.20, calmar=0.60, full_mdd=0.30)
    soft = _gate("soft", total=0.40, cagr=0.21, calmar=0.61, full_mdd=0.35)
    soft_equal = _gate("soft_equal", total=0.50, cagr=0.20, calmar=0.70, full_mdd=0.31)
    hard = _gate("hard", total=0.90, cagr=0.50, calmar=1.0, full_mdd=0.3500001)
    early_fail = _gate("early", total=1.0, full_mdd=0.9, dsr=0.89)

    result = select_alpha_max_prelock_champion([hard, soft_equal, early_fail, soft, normal])
    assert result.prelock_champion == "soft"
    assert result.selected_candidate_id == "soft"
    assert result.ranked_candidate_ids == ("soft", "normal")
    by_id = {decision.row_id: decision for decision in result.decisions}
    assert by_id["normal"].mdd_band == "normal"
    assert by_id["soft"].mdd_band == "soft"
    assert by_id["soft_equal"].rejection_reasons == (
        "soft_mdd_not_strictly_superior_to_best_normal",
    )
    assert by_id["hard"].rejection_reasons == ("mdd_above_hard_limit",)
    assert by_id["early"].evaluated_gates == ("dsr",)
    assert by_id["early"].rejection_reasons == ("dsr_below_threshold",)


def test_historical_ranking_is_report_only_and_terminal_precedence_is_singular() -> None:
    rows = [
        _gate("champion", role="historical_report", total=0.2),
        _gate("other", role="historical_report", total=0.3),
    ]
    result = rank_alpha_max_historical_report(rows)
    assert result.historical_evaluation_leader == "other"
    assert result.selected_candidate_id is None
    assert result.prelock_champion is None

    assert (
        alpha_max_terminal_outcome(
            None,
            champion_historical_complete=True,
            champion_historical_passed=True,
        )
        == "no_demonstrated_alpha"
    )
    assert (
        alpha_max_terminal_outcome(
            "champion",
            champion_historical_complete=False,
            champion_historical_passed=True,
        )
        == "historical_evaluation_incomplete"
    )
    assert (
        alpha_max_terminal_outcome(
            "champion",
            champion_historical_complete=True,
            champion_historical_passed=False,
        )
        == "prelock_champion_historical_robustness_failed"
    )
    assert (
        alpha_max_terminal_outcome(
            "champion",
            champion_historical_complete=True,
            champion_historical_passed=True,
        )
        == "prelock_champion_historical_robustness_passed"
    )
    terminal = build_alpha_max_terminal_state(
        prelock_champion="champion",
        champion_historical_complete=True,
        champion_historical_passed=True,
        historical_evaluation_leader="other",
        incumbent_comparison_status="unavailable",
    )
    assert terminal.selected_candidate_id == "champion"
    assert terminal.leader_differs_from_prelock_champion is True
    assert terminal.historical_exposure_status == "committed_period_outcomes_observed"
    assert terminal.requires_fresh_confirmation is True
    assert terminal.confirmation_status == "not_run"


def test_unavailable_row_cost_cells_and_prelock_inventory_are_immutable_canonical() -> None:
    cells = tuple(
        AlphaMaxCostCellEvidence.unavailable(
            row_id="incumbent",
            domain="validation",
            split_or_fold_id="validation_w01",
            nominal_cost_bps=cost,
            status="incumbent_replay_unavailable",
        )
        for cost in (30, 10, 20, 15)
    )
    row = AlphaMaxRowEvidence(
        row_id="incumbent",
        matrix_role="incumbent",
        status="incumbent_replay_unavailable",
        evidence_tier="identity",
        selection_valid=False,
        cost_cells=cells,
    )
    assert tuple(cell.nominal_cost_bps for cell in row.cost_cells) == (10, 15, 20, 30)
    assert canonical_alpha_max_row_bytes(row) == canonical_alpha_max_row_bytes(row)

    artifacts_a = {
        "rows/incumbent.json": canonical_alpha_max_row_bytes(row),
        "config.json": b"{}\n",
    }
    artifacts_b = dict(reversed(tuple(artifacts_a.items())))
    first = build_alpha_max_prelock_seal(
        artifacts_a,
        prelock_champion=None,
        selected_candidate_id=None,
    )
    second = build_alpha_max_prelock_seal(
        artifacts_b,
        prelock_champion=None,
        selected_candidate_id=None,
    )
    assert first == second
    assert first.sha256 == hashlib.sha256(first.canonical_bytes).hexdigest()
    with pytest.raises(FrozenInstanceError):
        first.sha256 = _HASH_A
    with pytest.raises(ValueError, match="prelock_artifact_path_invalid"):
        build_alpha_max_prelock_seal(
            {"../escape": b"x"},
            prelock_champion=None,
            selected_candidate_id=None,
        )
    with pytest.raises(ValueError, match="selection_identity_mismatch"):
        build_alpha_max_prelock_seal(
            artifacts_a,
            prelock_champion="a",
            selected_candidate_id="b",
        )
    with pytest.raises(ValueError, match="historical_input_forbidden"):
        build_alpha_max_prelock_seal(
            {"historical_evaluation/root-seal.json": b"poison"},
            prelock_champion=None,
            selected_candidate_id=None,
        )


def test_complete_row_cost_cells_bind_real_receipts_metrics_and_nominal_gate(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw-validation"
    feature_root = tmp_path / "feature-validation"
    raw_root.mkdir()
    feature_root.mkdir()
    start, end = evidence._ROOT_INTERVALS["validation"]
    raw_receipt = AlphaMaxRootReceipt(
        root_id="validation",
        root_kind="raw",
        path=str(raw_root),
        exchange="binance",
        start_utc=start,
        end_utc=end,
        inventory_sha256=_HASH_A,
        content_sha256=_HASH_B,
        file_count=1,
    )
    feature_receipt = AlphaMaxRootReceipt(
        root_id="validation",
        root_kind="feature",
        path=str(feature_root),
        exchange="binance",
        start_utc=start,
        end_utc=end,
        inventory_sha256="c" * 64,
        content_sha256="d" * 64,
        file_count=1,
    )
    capsule_path = tmp_path / "row-a.capsule.json"
    manifest_path = tmp_path / "row-a.json"
    capsule_bytes = b'{"capsule":true}\n'
    manifest_bytes = b'{"manifest":true}\n'
    capsule_path.write_bytes(capsule_bytes)
    manifest_path.write_bytes(manifest_bytes)
    capsule_receipt = AlphaMaxCapsuleReceipt(
        row_id="row-a",
        phase="validation_train_fit",
        prefix_id="validation_w01",
        path=str(capsule_path),
        sha256=hashlib.sha256(capsule_bytes).hexdigest(),
        byte_count=len(capsule_bytes),
    )
    manifest_receipt = AlphaMaxManifestReceipt(
        row_id="row-a",
        phase="validation_train_fit",
        path=str(manifest_path),
        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        byte_count=len(manifest_bytes),
    )
    calendar = tuple(start + timedelta(hours=4 * index) for index in range(6))
    equities = (10_100.0, 10_000.0, 10_200.0, 10_150.0, 10_300.0, 10_400.0)
    stream = build_alpha_max_primary_return_stream(
        tuple(
            AlphaMaxEquityEndpoint(timestamp=timestamp, equity=equity)
            for timestamp, equity in zip(calendar, equities, strict=True)
        ),
        calendar,
    )
    metrics = compute_alpha_max_metric_statistics(stream, equities)
    reconciliation = reconcile_alpha_max_cost_attribution(
        (),
        (),
        (),
        (),
        portfolio_fee_total=0.0,
        portfolio_funding_total=0.0,
    )
    gate = AlphaMaxGateInput(
        row_id="row-a",
        comparison_role="prelock_selection",
        evidence_tier="actual_engine",
        comparison_valid=True,
        nominal_cost_bps=30,
        cumulative_return=metrics.canonical_metrics["total_return"],
        cagr=metrics.canonical_metrics["cagr"],
        calmar=metrics.canonical_metrics["calmar"],
        net_sharpe=metrics.canonical_metrics["sharpe"],
        full_event_mdd=metrics.full_event_mdd,
        reporting_4h_mdd=metrics.reporting_4h_mdd,
        dsr=0.95,
        spa_pvalue=0.01,
        pbo=0.10,
        native_data_coverage_complete=True,
        funding_coverage_complete=True,
        hash_valid=True,
        manifest_valid=True,
        reconciliation_complete=True,
        ruin=False,
        raw_root_set_sha256=_HASH_A,
        feature_root_set_sha256=_HASH_B,
        universe_sha256="c" * 64,
        calendar_sha256=stream.calendar_sha256,
        seed_schedule_sha256="e" * 64,
    )
    cells = tuple(
        AlphaMaxCostCellEvidence(
            row_id="row-a",
            domain="validation",
            split_or_fold_id="validation_w01",
            nominal_cost_bps=cost,
            status="complete",
            evidence_tier="actual_engine",
            selection_valid=True,
            seed=alpha_max_common_rng_seed("validation_w01", cost),
            raw_root_receipts=(raw_receipt,),
            feature_root_receipts=(feature_receipt,),
            capsule_receipt=capsule_receipt,
            manifest_receipt=manifest_receipt,
            primary_return_stream=stream,
            metric_statistics=metrics,
            reconciliation=reconciliation,
            gate_input=gate if cost == 30 else None,
            runtime_contract_sha256="1" * 64,
            config_sha256="2" * 64,
            coverage_sha256="3" * 64,
            exposure_sha256="4" * 64,
            ruin=False,
        )
        for cost in (10, 15, 20, 30)
    )
    row = AlphaMaxRowEvidence(
        row_id="row-a",
        matrix_role="component",
        status="complete",
        evidence_tier="actual_engine",
        selection_valid=True,
        cost_cells=cells,
    )

    assert row.gate_input is gate
    assert canonical_alpha_max_cost_cell_bytes(cells[-1]).endswith(b"\n")
    assert json.loads(canonical_alpha_max_row_bytes(row))["row_id"] == "row-a"


def test_manifest_receipt_binds_materialized_file_stem_and_exact_bytes(tmp_path: Path) -> None:
    raw = b'{"children":[{"candidate_id":"component"}]}\n'
    path = tmp_path / "row-a.json"
    path.write_bytes(raw)
    materialization = AlphaMaxManifestMaterialization(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        canonical_bytes=raw,
        strategy_params={},
    )

    receipt = AlphaMaxManifestReceipt.from_materialization(
        materialization,
        phase="validation_train_fit",
    )

    assert receipt.row_id == "row-a"
    assert receipt.sha256 == materialization.sha256
    assert receipt.byte_count == len(raw)


def test_streaming_full_event_tracker_is_exact_and_constant_memory_for_large_stream() -> None:
    tracker = AlphaMaxStreamingEquityTracker(initial_capital=10_000.0)
    values: list[float] = []
    peak = 10_000.0
    current_duration = 0
    expected_duration = 0
    start = datetime(2025, 1, 1, tzinfo=UTC)
    for index in range(120_000):
        value = 10_000.0 + 750.0 * math.sin(index / 137.0) + (index % 11)
        values.append(value)
        timestamp = start + timedelta(seconds=index)
        tracker.observe((timestamp.timestamp(), value))
        if value >= peak:
            peak = value
            current_duration = 0
        else:
            current_duration += 1
            expected_duration = max(expected_duration, current_duration)

    snapshot = tracker.finalize()
    assert snapshot.full_event_mdd == pytest.approx(alpha_max_full_event_mdd(values))
    assert snapshot.max_drawdown_duration_events == expected_duration
    assert snapshot.event_count == len(values)
    assert snapshot.last_timestamp_ms == int(
        (start + timedelta(seconds=119_999)).timestamp() * 1000
    )
    assert tracker.retained_point_count == 0
    assert tracker.state_size_bytes < 4096
    with pytest.raises(FrozenInstanceError):
        snapshot.event_count = 1


def test_streaming_full_event_tracker_rejects_malformed_portfolio_sink_points() -> None:
    tracker = AlphaMaxStreamingEquityTracker()

    with pytest.raises(TypeError, match="streaming_equity_point_invalid"):
        tracker.observe([1.0, 10_000.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="streaming_equity_unix_seconds_invalid"):
        tracker.observe((-1.0, 10_000.0))
    with pytest.raises(ValueError, match="streaming_equity_unix_seconds_invalid"):
        tracker.observe((1e308, 10_000.0))
