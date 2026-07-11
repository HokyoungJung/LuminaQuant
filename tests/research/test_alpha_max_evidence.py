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
    AlphaMaxTerminalGateEvidence,
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
    build_alpha_max_normalized_fold_segment_evidence,
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
    validate_alpha_max_admitted_symbols,
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


def _write_sparse_raw_root(root: Path, root_id: str) -> None:
    start, end = evidence._ROOT_INTERVALS[root_id]
    month = start.replace(day=1)
    months: list[datetime] = []
    while month < end:
        months.append(month)
        month = (
            month.replace(year=month.year + 1, month=1)
            if month.month == 12
            else month.replace(month=month.month + 1)
        )
    for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
        directory = root / "market_ohlcv_1s" / "binance" / symbol
        directory.mkdir(parents=True, exist_ok=True)
        for partition_start in months:
            partition_end = (
                partition_start.replace(year=partition_start.year + 1, month=1)
                if partition_start.month == 12
                else partition_start.replace(month=partition_start.month + 1)
            )
            owned_start = max(start, partition_start)
            owned_end = min(end, partition_end)
            timestamps = [owned_start + timedelta(seconds=7), owned_end - timedelta(seconds=11)]
            pl = pytest.importorskip("polars")
            pl.DataFrame(
                {
                    "datetime": timestamps,
                    "symbol": [symbol, symbol],
                    "exchange": ["binance", "binance"],
                    "close": [100.0, 101.0],
                    "volume": [1.0, 2.0],
                }
            ).write_parquet(directory / f"{partition_start:%Y-%m}.parquet")


def _write_feature_root(root: Path, root_id: str) -> None:
    pl = pytest.importorskip("polars")
    start, end = evidence._ROOT_INTERVALS[root_id]
    day = start
    while day < end:
        for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
            directory = (
                root
                / "feature_points"
                / "exchange=binance"
                / f"symbol={symbol}"
                / f"date={day:%Y-%m-%d}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            timestamps = [
                int((day + timedelta(hours=hour)).timestamp() * 1000) for hour in (0, 8, 16)
            ]
            pl.DataFrame(
                {
                    "timestamp_ms": timestamps,
                    "funding_rate": [0.0001, -0.0002, 0.0003],
                    "symbol": [symbol] * 3,
                    "exchange": ["binance"] * 3,
                }
            ).write_parquet(directory / "part-0.parquet")
        day += timedelta(days=1)


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
    _write_sparse_raw_root(root, "purge")

    first = seal_alpha_max_root_tree("purge", "raw", root)
    second = seal_alpha_max_root_tree("purge", "raw", root)
    assert first == second
    assert first.symbols == ALPHA_MAX_CANDIDATE_SYMBOLS
    assert len(first.entries) == len(ALPHA_MAX_CANDIDATE_SYMBOLS)
    start, end = evidence._ROOT_INTERVALS["purge"]
    assert min(entry.minimum_timestamp_ms for entry in first.entries) > int(
        start.timestamp() * 1000
    )
    assert max(entry.maximum_timestamp_ms for entry in first.entries) < int(
        (end - timedelta(seconds=1)).timestamp() * 1000
    )
    assert first.inventory_sha256 == second.inventory_sha256
    assert first.content_sha256 == second.content_sha256
    assert first.canonical_bytes == second.canonical_bytes
    assert first.to_receipt().content_sha256 == first.content_sha256

    target = root / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "2025-06.parquet"
    pl = pytest.importorskip("polars")
    pl.DataFrame(
        {
            "datetime": [start + timedelta(seconds=9), end - timedelta(seconds=13)],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "exchange": ["binance", "binance"],
            "close": [100.0, 105.0],
            "volume": [1.0, 2.0],
        }
    ).write_parquet(target)
    mutated = seal_alpha_max_root_tree("purge", "raw", root)
    assert mutated.content_sha256 != first.content_sha256
    assert mutated.inventory_sha256 != first.inventory_sha256

    (root / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        seal_alpha_max_root_tree("purge", "raw", root)
    with pytest.raises(ValueError, match="must_be_absolute"):
        seal_alpha_max_root_tree("train", "raw", Path("relative"))


def test_actual_run_domain_seals_bind_current_raw_and_adjacent_features(
    tmp_path: Path,
) -> None:
    def seal_stub(root_id: str, root_kind: str):
        path = (tmp_path / f"{root_id}-{root_kind}").resolve()
        path.mkdir()
        start, end = evidence._ROOT_INTERVALS[root_id]
        value = object.__new__(evidence.AlphaMaxRootSeal)
        fields = {
            "root_id": root_id,
            "root_kind": root_kind,
            "path": str(path),
            "exchange": "binance",
            "symbols": ALPHA_MAX_CANDIDATE_SYMBOLS,
            "start_utc": start,
            "end_utc": end,
            "entries": (object(),),
            "inventory_sha256": _HASH_A,
            "content_sha256": _HASH_B,
            "canonical_bytes": b"unused-in-domain-seal-test\n",
            "sha256": "c" * 64,
        }
        for name, field_value in fields.items():
            object.__setattr__(value, name, field_value)
        return value

    purge_raw = seal_stub("purge", "raw")
    validation_raw = seal_stub("validation", "raw")
    purge_feature = seal_stub("purge", "feature")
    validation_feature = seal_stub("validation", "feature")

    raw_receipts = evidence._alpha_max_validate_domain_root_seals(
        (validation_raw,),
        domain="validation",
        root_kind="raw",
    )
    feature_receipts = evidence._alpha_max_validate_domain_root_seals(
        (purge_feature, validation_feature),
        domain="validation",
        root_kind="feature",
    )

    assert tuple(receipt.root_id for receipt in raw_receipts) == ("validation",)
    assert tuple(receipt.root_id for receipt in feature_receipts) == (
        "purge",
        "validation",
    )
    with pytest.raises(ValueError, match="root_domain_mismatch"):
        evidence._alpha_max_validate_domain_root_seals(
            (purge_raw, validation_raw),
            domain="validation",
            root_kind="raw",
        )
    with pytest.raises(ValueError, match="root_domain_mismatch"):
        evidence._alpha_max_validate_domain_root_seals(
            (validation_feature,),
            domain="validation",
            root_kind="feature",
        )


@pytest.mark.parametrize(
    "timestamps,match",
    [
        (
            lambda start, end: [start + timedelta(seconds=7)] * 2,
            "duplicate_or_null",
        ),
        (
            lambda start, end: [end - timedelta(seconds=11), start + timedelta(seconds=7)],
            "not_strictly_increasing",
        ),
        (
            lambda start, end: [start - timedelta(seconds=1), end - timedelta(seconds=11)],
            "outside_interval|partition_content",
        ),
    ],
)
def test_raw_root_rejects_duplicate_nonmonotone_and_out_of_range_rows(
    tmp_path: Path,
    timestamps,
    match: str,
) -> None:
    root = tmp_path / "raw"
    _write_sparse_raw_root(root, "purge")
    start, end = evidence._ROOT_INTERVALS["purge"]
    target = root / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "2025-06.parquet"
    pl = pytest.importorskip("polars")
    pl.DataFrame(
        {
            "datetime": timestamps(start, end),
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "exchange": ["binance", "binance"],
            "close": [100.0, 101.0],
            "volume": [1.0, 2.0],
        }
    ).write_parquet(target)
    with pytest.raises(ValueError, match=match):
        seal_alpha_max_root_tree("purge", "raw", root)


def test_feature_root_binds_content_ownership_and_every_funding_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    _write_feature_root(root, "purge")
    seal = seal_alpha_max_root_tree("purge", "feature", root)
    assert seal.symbols == ALPHA_MAX_CANDIDATE_SYMBOLS
    assert len(seal.entries) == 7 * len(ALPHA_MAX_CANDIDATE_SYMBOLS)

    target = (
        root
        / "feature_points"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "date=2025-06-01"
        / "part-0.parquet"
    )
    pl = pytest.importorskip("polars")
    frame = pl.read_parquet(target)
    frame.with_columns(pl.lit("ETHUSDT").alias("symbol")).write_parquet(target)
    with pytest.raises(ValueError, match="content_symbol_mismatch"):
        seal_alpha_max_root_tree("purge", "feature", root)

    frame.filter(pl.col("timestamp_ms") != frame["timestamp_ms"][1]).write_parquet(target)
    with pytest.raises(ValueError, match=r"funding_boundary_missing|timestamp_cadence"):
        seal_alpha_max_root_tree("purge", "feature", root)


def test_feature_root_accepts_causal_as_of_points_before_funding_boundaries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    _write_feature_root(root, "purge")
    target = (
        root
        / "feature_points"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "date=2025-06-01"
        / "part-0.parquet"
    )
    pl = pytest.importorskip("polars")
    frame = pl.read_parquet(target)
    timestamps = frame.get_column("timestamp_ms").to_list()
    frame.with_columns(
        pl.Series(
            "timestamp_ms",
            [timestamps[0], timestamps[1] - 1000, timestamps[2] - 1000],
        )
    ).write_parquet(target)

    seal = seal_alpha_max_root_tree("purge", "feature", root)

    entry = next(
        row for row in seal.entries if "symbol=BTCUSDT/date=2025-06-01" in row.relative_path
    )
    assert entry.maximum_gap_ms == evidence._FUNDING_INTERVAL_MS


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
    prelock = select_alpha_max_prelock_champion(
        [_gate("champion", role="prelock_selection", total=0.2)]
    )
    terminal = build_alpha_max_terminal_state(
        prelock_selection=prelock,
        champion_historical_nominal_30_cell=None,
        historical_ranking=result,
        incumbent_comparison_status="unavailable",
    )
    assert terminal.selected_candidate_id == "champion"
    assert terminal.leader_differs_from_prelock_champion is True
    assert terminal.terminal_outcome == "historical_evaluation_incomplete"
    assert terminal.historical_exposure_status == "historical_evaluation_incomplete"
    assert terminal.requires_fresh_confirmation is True
    assert terminal.confirmation_status == "not_run"

    no_champion = select_alpha_max_prelock_champion(
        [_gate("rejected", role="prelock_selection", dsr=0.10)]
    )
    report_only_terminal = build_alpha_max_terminal_state(
        prelock_selection=no_champion,
        champion_historical_nominal_30_cell=None,
        historical_ranking=result,
        incumbent_comparison_status="unavailable",
    )
    assert report_only_terminal.prelock_champion is None
    assert report_only_terminal.selected_candidate_id is None
    assert report_only_terminal.historical_evaluation_leader == "other"
    assert report_only_terminal.terminal_outcome == "no_demonstrated_alpha"
    assert report_only_terminal.historical_exposure_status == "committed_period_outcomes_observed"


def test_unavailable_row_cost_cells_and_prelock_inventory_are_immutable_canonical() -> None:
    cells = tuple(
        AlphaMaxCostCellEvidence.unavailable(
            row_id="incumbent",
            domain="validation",
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


def test_capsule_receipt_parses_causal_envelope_and_rejects_fold_relabel(
    tmp_path: Path,
) -> None:
    manifest_sha256 = "c" * 64
    capsule_scope = {"ready": True}
    capsule_sha256 = hashlib.sha256(
        json.dumps(capsule_scope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    state_payload = {
        "capsule": {**capsule_scope, "sha256": capsule_sha256},
        "capsule_sha256": capsule_sha256,
        "discarded_signal_count": 3,
        "fill_event_count": 0,
        "finalized_children": {"component": {"ready": True}},
        "funding_event_count": 0,
        "manifest_sha256": manifest_sha256,
        "market_event_count": 0,
        "order_event_count": 0,
        "phase_id": "purge",
        "portfolio_mode": "manifest:/stable/row-a.json",
        "trade_count": 0,
        "windows_processed": 10,
    }
    raw = AlphaMaxCapsuleReceipt.canonical_envelope_bytes(
        row_id="row-a",
        phase="validation_train_fit",
        prefix_id="validation_w01",
        manifest_sha256=manifest_sha256,
        state_payload=state_payload,
    )
    path = tmp_path / "capsule.json"
    path.write_bytes(raw)
    receipt = AlphaMaxCapsuleReceipt.from_path(
        path,
        row_id="row-a",
        phase="validation_train_fit",
        prefix_id="validation_w01",
        manifest_sha256=manifest_sha256,
        relative_path="capsules/row-a/validation_w01.json",
    )
    assert receipt.capsule_phase_id == "purge"
    assert receipt.boundary_utc == evidence._ALPHA_MAX_FOLD_INTERVALS["validation_w01"][0]
    with pytest.raises(ValueError, match="envelope_scope_mismatch"):
        AlphaMaxCapsuleReceipt.from_path(
            path,
            row_id="row-a",
            phase="validation_train_fit",
            prefix_id="validation_w02",
            manifest_sha256=manifest_sha256,
            relative_path="capsules/row-a/validation_w02.json",
        )


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
        relative_path="manifests/validation_train_fit/row-a.json",
    )

    assert receipt.row_id == "row-a"
    assert receipt.sha256 == materialization.sha256
    assert receipt.byte_count == len(raw)


def test_effective_runtime_config_binds_all_attributes_and_rejects_seed_forge() -> None:
    admitted = validate_alpha_max_admitted_symbols(
        ALPHA_MAX_CANDIDATE_SYMBOLS,
        ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
    )
    fold_id = "validation_w01"
    start, end = evidence._ALPHA_MAX_FOLD_INTERVALS[fold_id]
    static = {
        "DECISION_CADENCE_SECONDS": 1,
        "INITIAL_CAPITAL": 10_000.0,
        "STATIC_SENTINEL": "sealed",
    }
    payload = {
        **static,
        "END_DATE": end.isoformat().replace("+00:00", "Z"),
        "RANDOM_SEED": alpha_max_common_rng_seed(fold_id, 30),
        "SLIPPAGE_RATE": 0.0025,
        "START_DATE": start.isoformat().replace("+00:00", "Z"),
        "SYMBOLS": list(admitted),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    runtime = {
        "attribute_allowlist": sorted(payload),
        "static_attributes": static,
    }
    assert (
        evidence._alpha_max_validate_effective_config_bytes(
            raw,
            hashlib.sha256(raw).hexdigest(),
            split_or_fold_id=fold_id,
            nominal_cost_bps=30,
            admitted_symbols=admitted,
            runtime_contract_payload=runtime,
        )
        == payload
    )
    poisoned = {**payload, "RANDOM_SEED": payload["RANDOM_SEED"] + 1}
    poisoned_raw = json.dumps(poisoned, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="runtime_binding_mismatch"):
        evidence._alpha_max_validate_effective_config_bytes(
            poisoned_raw,
            hashlib.sha256(poisoned_raw).hexdigest(),
            split_or_fold_id=fold_id,
            nominal_cost_bps=30,
            admitted_symbols=admitted,
            runtime_contract_payload=runtime,
        )


def _fake_fold_runs_and_live_segments(domain: str):
    fold_runs = []
    segment_inputs = []
    aggregate = AlphaMaxStreamingEquityTracker()
    current = 10_000.0
    for index, fold_id in enumerate(evidence._ALPHA_MAX_DOMAIN_FOLD_IDS[domain]):
        start, _ = evidence._ALPHA_MAX_FOLD_INTERVALS[fold_id]
        timestamps = (
            int((start + timedelta(seconds=1)).timestamp() * 1000),
            int((start + timedelta(seconds=2)).timestamp() * 1000),
        )
        source_values = (8_000.0, 20_000.0) if index == 0 else (10_000.0, 10_000.0)
        source_tracker = AlphaMaxStreamingEquityTracker()
        for timestamp, value in zip(timestamps, source_values, strict=True):
            source_tracker.update(value, timestamp)
        source = source_tracker.finalize()
        run = object.__new__(evidence.AlphaMaxActualEngineRunReceipt)
        object.__setattr__(run, "domain", domain)
        object.__setattr__(run, "split_or_fold_id", fold_id)
        object.__setattr__(run, "full_event_equity", source)
        object.__setattr__(run, "liquidation_event_count", 0)
        object.__setattr__(run, "sha256", hashlib.sha256(f"run:{fold_id}".encode()).hexdigest())
        fold = object.__new__(evidence.AlphaMaxFoldRunEvidence)
        object.__setattr__(fold, "actual_engine_run", run)
        object.__setattr__(fold, "status", "complete")
        object.__setattr__(fold, "sha256", hashlib.sha256(f"fold:{fold_id}".encode()).hexdigest())
        fold_runs.append(fold)

        scale = current / 10_000.0
        normalized_values = tuple(scale * value for value in source_values)
        for timestamp, value in zip(timestamps, normalized_values, strict=True):
            aggregate.update(value, timestamp)
        normalized_end = normalized_values[-1]
        segment_inputs.append(
            build_alpha_max_normalized_fold_segment_evidence(
                fold_id=fold_id,
                source_streaming_equity_sha256=source.sha256,
                source_event_stream_sha256=source.event_stream_sha256,
                normalization_scale=scale,
                normalized_starting_equity=current,
                normalized_ending_equity=normalized_end,
                normalized_segment_event_stream_sha256=hashlib.sha256(
                    f"normalized:{fold_id}".encode()
                ).hexdigest(),
                event_count=source.event_count,
                first_timestamp_ms=source.first_timestamp_ms,
                last_timestamp_ms=source.last_timestamp_ms,
                aggregate_prefix_event_count=aggregate.event_count,
                aggregate_prefix_event_stream_sha256=aggregate.event_stream_sha256,
            )
        )
        current = normalized_end
    return tuple(fold_runs), aggregate.finalize(), tuple(segment_inputs)


def test_live_combined_stream_preserves_peak_after_min_chronology_and_hash_binding() -> None:
    fold_runs, live, segments = _fake_fold_runs_and_live_segments("validation")
    combined = evidence._build_alpha_max_combined_streaming_equity(
        fold_runs,
        live,
        segments,
    )
    assert len(combined.fold_ids) == 12
    assert combined.full_event_mdd == pytest.approx(0.20)
    assert evidence._ALPHA_MAX_DOMAIN_ENGINE_RUN_COUNT == {
        "validation": 816,
        "historical_exposed_evaluation": 680,
    }
    assert len(evidence._ALPHA_MAX_DOMAIN_FOLD_IDS["historical_exposed_evaluation"]) == 10

    fake = AlphaMaxStreamingEquityTracker()
    for index, segment in enumerate(segments):
        fake.update(5_000.0 if index == 0 else 20_000.0, segment.first_timestamp_ms)
        fake.update(20_000.0, segment.last_timestamp_ms)
    forged = fake.finalize()
    assert forged.event_count == live.event_count
    assert forged.ending_equity == live.ending_equity
    with pytest.raises(ValueError, match="live_binding_mismatch"):
        evidence._build_alpha_max_combined_streaming_equity(
            fold_runs,
            forged,
            segments,
        )


def test_typed_ruin_gate_is_rejected_without_fabricated_metrics() -> None:
    terminal = AlphaMaxTerminalGateEvidence(
        row_id="ruined",
        comparison_role="prelock_selection",
        domain="validation",
        nominal_cost_bps=30,
        pre_gate_evidence_sha256="1" * 64,
        fold_run_set_sha256="2" * 64,
        ruined_fold_ids=("validation_w01", "validation_w02"),
        streaming_ruin_fold_ids=("validation_w01",),
        liquidation_fold_ids=("validation_w02",),
        raw_root_set_sha256="3" * 64,
        feature_root_set_sha256="4" * 64,
        universe_sha256="5" * 64,
        seed_schedule_sha256="6" * 64,
    )
    result = select_alpha_max_prelock_champion((terminal,))
    assert result.prelock_champion is None
    assert result.decisions[0].rejection_reasons == ("ruin_detected",)
    assert result.decisions[0].gate_mdd is None


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


def test_streaming_full_event_tracker_records_zero_and_negative_equity_as_ruin() -> None:
    tracker = AlphaMaxStreamingEquityTracker()
    tracker.observe((1.0, 10_000.0))
    tracker.observe((2.0, 0.0))
    tracker.observe((3.0, -25.0))

    snapshot = tracker.finalize()

    assert snapshot.ruin is True
    assert snapshot.ending_equity == -25.0
    assert snapshot.full_event_mdd == 1.0
    assert snapshot.uncapped_full_event_drawdown == pytest.approx(1.0025)
