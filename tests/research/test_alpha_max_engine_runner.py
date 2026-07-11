from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import lumina_quant.research.alpha_max_engine_runner as alpha_max_runner

from lumina_quant.research.alpha_max_engine_runner import (
    ALPHA_MAX_COST_CELL_BPS,
    AlphaMaxAttributionCollector,
    AlphaMaxRuntimeContractError,
    construct_alpha_max_engine,
    create_alpha_max_historical_package,
    create_alpha_max_prelock_bundle,
    orchestrate_alpha_max_status_matrix,
    preflight_alpha_max_runtime_contract,
    seal_alpha_max_manifest_activation,
    validate_alpha_max_engine_activation,
)
from lumina_quant.research.alpha_max_evidence import (
    AlphaMaxActualEngineRunReceipt,
    AlphaMaxCapsuleReceipt,
    AlphaMaxCostCellEvidence,
    AlphaMaxCostCellPreGateEvidence,
    AlphaMaxFoldRunEvidence,
    AlphaMaxFundingBoundaryResolver,
    AlphaMaxManifestReceipt,
    AlphaMaxOrderedFundingLookup,
    AlphaMaxRootSeal,
    AlphaMaxStreamingEquityTracker,
    FeatureRootSpec,
    materialize_alpha_max_manifest,
)
from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.strategies.artifact_portfolio_mode import ArtifactPortfolioModeStrategy


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (REPO_ROOT / "configs/research/alpha_max_portfolio_20260710.json").resolve()


@pytest.fixture(autouse=True)
def _clean_lq_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("LQ_"):
            monkeypatch.delenv(key, raising=False)


def _nodes() -> list[dict[str, object]]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return payload["current_trial_registry"]["nodes"]


def _owned_root(tmp_path: Path) -> Path:
    root = (tmp_path / "run").resolve()
    (root / "manifests/validation_train_fit").mkdir(parents=True)
    (root / "manifests/prelock_final_refit").mkdir()
    return root


def test_status_matrix_has_84_statuses_68_logical_cells_and_816_fold_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    manifests: dict[str, AlphaMaxManifestReceipt] = {}
    capsules: dict[tuple[str, str], AlphaMaxCapsuleReceipt] = {}

    monkeypatch.setattr(
        "lumina_quant.research.alpha_max_engine_runner.canonical_alpha_max_cost_cell_bytes",
        lambda _value: b"typed-cell\n",
    )

    def bare(cls, **fields):
        value = object.__new__(cls)
        for name, field_value in fields.items():
            object.__setattr__(value, name, field_value)
        return value

    def execute(row, nominal_cost_bps):
        row_id = row["row_id"]
        calls.append((row_id, nominal_cost_bps))
        manifest = manifests.setdefault(
            row_id,
            bare(
                AlphaMaxManifestReceipt,
                sha256=f"{abs(hash(('manifest', row_id))):064x}"[-64:],
            ),
        )
        fold_runs = []
        for index in range(1, 13):
            fold_id = f"validation_w{index:02d}"
            capsule = capsules.setdefault(
                (row_id, fold_id),
                bare(
                    AlphaMaxCapsuleReceipt,
                    prefix_id=fold_id,
                    sha256=f"{abs(hash((row_id, fold_id))):064x}"[-64:],
                ),
            )
            actual = bare(
                AlphaMaxActualEngineRunReceipt,
                split_or_fold_id=fold_id,
                capsule_receipt=capsule,
                manifest_receipt=manifest,
            )
            fold_runs.append(
                bare(
                    AlphaMaxFoldRunEvidence,
                    actual_engine_run=actual,
                )
            )
        pre_gate = bare(
            AlphaMaxCostCellPreGateEvidence,
            fold_runs=tuple(fold_runs),
        )
        return bare(
            AlphaMaxCostCellEvidence,
            row_id=row_id,
            domain="validation",
            nominal_cost_bps=nominal_cost_bps,
            status="complete",
            evidence_tier="actual_engine",
            selection_valid=True,
            pre_gate_evidence=pre_gate,
        )

    result = orchestrate_alpha_max_status_matrix(_nodes(), execute)

    assert len(result.statuses) == 21 * 4
    assert result.engine_cell_count == 17 * 4
    assert len(calls) == 17 * 4
    assert (
        sum(
            len(status.evidence.pre_gate_evidence.fold_runs)
            for status in result.statuses
            if status.engine_constructed
        )
        == 816
    )
    assert {cost for _, cost in calls} == set(ALPHA_MAX_COST_CELL_BPS)
    assert not any(row_id.startswith("incumbent_") for row_id, _ in calls)
    assert not any(row_id.startswith("diagnostic_") for row_id, _ in calls)


def test_production_physical_fold_schedules_are_exact() -> None:
    validation = alpha_max_runner._alpha_max_physical_fold_schedule("validation")
    historical = alpha_max_runner._alpha_max_physical_fold_schedule("historical_exposed_evaluation")

    assert len(validation) == 816
    assert len(historical) == 680
    assert len(set(validation)) == 816
    assert len(set(historical)) == 680
    assert all(
        [fold_id for row, cost, fold_id in validation if row == row_id and cost == nominal]
        == [f"validation_w{index:02d}" for index in range(1, 13)]
        for row_id in alpha_max_runner._ALPHA_MAX_RESOLVABLE_ROWS
        for nominal in ALPHA_MAX_COST_CELL_BPS
    )
    alpha_max_runner._validate_alpha_max_physical_fold_schedule(
        validation,
        domain="validation",
    )
    alpha_max_runner._validate_alpha_max_physical_fold_schedule(
        historical,
        domain="historical_exposed_evaluation",
    )
    for invalid, domain in (
        (validation[:-1], "validation"),
        ((*validation, validation[-1]), "validation"),
        (historical[:-1], "historical_exposed_evaluation"),
        ((*historical, historical[-1]), "historical_exposed_evaluation"),
    ):
        with pytest.raises(
            AlphaMaxRuntimeContractError,
            match="physical_fold_cardinality_mismatch",
        ):
            alpha_max_runner._validate_alpha_max_physical_fold_schedule(
                invalid,
                domain=domain,
            )
    assert all(
        len([fold_id for row, cost, fold_id in historical if row == row_id and cost == nominal])
        == 10
        for row_id in alpha_max_runner._ALPHA_MAX_RESOLVABLE_ROWS
        for nominal in ALPHA_MAX_COST_CELL_BPS
    )


def test_adjacent_feature_root_gap_is_closed_across_split() -> None:
    boundary = datetime(2025, 1, 1, tzinfo=UTC)

    def seal(root_id: str, *, first: int, last: int):
        entries = tuple(
            SimpleNamespace(
                relative_path=f"symbol={symbol}/date=2025-01-01/part.parquet",
                minimum_timestamp_ms=first,
                maximum_timestamp_ms=last,
            )
            for symbol in alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS
        )
        return SimpleNamespace(
            root_id=root_id,
            exchange="binance",
            entries=entries,
            start_utc=boundary if root_id == "right" else boundary.replace(year=2024),
            end_utc=boundary.replace(year=2026) if root_id == "right" else boundary,
        )

    predecessor = seal("left", first=1, last=10_000)
    current = seal("right", first=10_000 + 28_801_000, last=40_000_000)
    seals = {("left", "feature"): predecessor, ("right", "feature"): current}
    alpha_max_runner._validate_alpha_max_adjacent_feature_roots(
        seals,
        (("left", "right"),),
    )

    current.entries = tuple(
        SimpleNamespace(
            relative_path=entry.relative_path,
            minimum_timestamp_ms=10_000 + 28_801_001,
            maximum_timestamp_ms=entry.maximum_timestamp_ms,
        )
        for entry in current.entries
    )
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="adjacent_feature_root_funding_coverage_incomplete",
    ):
        alpha_max_runner._validate_alpha_max_adjacent_feature_roots(
            seals,
            (("left", "right"),),
        )


def test_runtime_read_audit_keeps_different_daily_paths_in_chronological_order() -> None:
    first = ("START_DATE", "END_DATE", "SYMBOLS")
    second = ("START_DATE", "SLIPPAGE_RATE", "END_DATE")

    retained, first_sha = alpha_max_runner._alpha_max_append_runtime_read_audit(
        None,
        first,
    )
    combined, combined_sha = alpha_max_runner._alpha_max_append_runtime_read_audit(
        retained,
        second,
    )

    assert retained == first
    assert combined == (*first, *second)
    assert first_sha != combined_sha
    assert combined_sha == alpha_max_runner._sha256(
        alpha_max_runner._canonical_bytes(list(combined))
    )
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="runtime_read_audit_invalid",
    ):
        alpha_max_runner._alpha_max_append_runtime_read_audit(
            combined,
            ("UNSEALED_FIELD",),
        )


def test_run_owned_root_writes_sealed_last_and_never_seals_mismatched_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (tmp_path / "sealed-last").resolve()
    root = alpha_max_runner._create_alpha_max_run_owned_root(output)
    writes: list[str] = []
    original_write = alpha_max_runner._write_bundle_file

    def recording_write(bundle_root, relative_path, payload):
        writes.append(relative_path)
        return original_write(bundle_root, relative_path, payload)

    monkeypatch.setattr(alpha_max_runner, "_write_bundle_file", recording_write)
    bundle = alpha_max_runner._finalize_alpha_max_run_owned_root(
        root,
        {"report/result.json": b"{}\n"},
        seal_bytes=b'{"sealed":true}\n',
    )
    assert writes[-1] == "SEALED.json"
    assert Path(bundle.seal_path).read_bytes() == b'{"sealed":true}\n'

    mismatch_root = alpha_max_runner._create_alpha_max_run_owned_root(
        (tmp_path / "mismatch").resolve()
    )
    original_write(mismatch_root, "report/result.json", b"first\n")
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="run_owned_artifact_mismatch",
    ):
        alpha_max_runner._finalize_alpha_max_run_owned_root(
            mismatch_root,
            {"report/result.json": b"second\n"},
            seal_bytes=b'{"sealed":true}\n',
        )
    assert not (mismatch_root / "SEALED.json").exists()
    alpha_max_runner._cleanup_partial_bundle(mismatch_root)
    assert not mismatch_root.exists()


def test_manifest_seal_rejects_hard_linked_target(tmp_path: Path) -> None:
    preflight = preflight_alpha_max_runtime_contract(str(CONFIG_PATH))
    admitted = preflight.candidate_symbols[:5]
    root = _owned_root(tmp_path)
    node = next(row for row in _nodes() if row["row_id"] == "component_trend_1x")
    materialized = materialize_alpha_max_manifest(
        node,
        {"component_trend_1x": 1.0},
        1.0,
        "validation_train_fit",
        str(CONFIG_PATH),
        str(root),
        preflight.candidate_symbols,
        admitted,
        "d" * 64,
    )
    seal = seal_alpha_max_manifest_activation(
        preflight,
        output_root=str(root),
        phase="validation_train_fit",
        manifest_path=materialized.path,
        admitted_symbols=admitted,
    )
    assert seal.manifest_receipt.sha256 == materialized.sha256

    target = Path(materialized.path)
    linked = target.with_suffix(".linked")
    os.link(target, linked)
    with pytest.raises(
        AlphaMaxRuntimeContractError, match="portfolio_manifest_activation_mismatch"
    ):
        seal_alpha_max_manifest_activation(
            preflight,
            output_root=str(root),
            phase="validation_train_fit",
            manifest_path=materialized.path,
            admitted_symbols=admitted,
        )


def test_actual_backtest_construction_binds_all_alpha_runtime_seams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = preflight_alpha_max_runtime_contract(str(CONFIG_PATH))
    admitted = preflight.candidate_symbols[:5]
    root = _owned_root(tmp_path)
    node = next(row for row in _nodes() if row["row_id"] == "component_trend_1x")
    materialized = materialize_alpha_max_manifest(
        node,
        {"component_trend_1x": 1.0},
        1.0,
        "validation_train_fit",
        str(CONFIG_PATH),
        str(root),
        preflight.candidate_symbols,
        admitted,
        "d" * 64,
    )
    purge_feature_root = (tmp_path / "features-purge").resolve()
    validation_feature_root = (tmp_path / "features-validation").resolve()
    purge_feature_root.mkdir()
    validation_feature_root.mkdir()
    purge_window = preflight.phase_windows["purge"]
    validation_window = preflight.phase_windows["validation"]
    lookup = AlphaMaxOrderedFundingLookup(
        (
            FeatureRootSpec(
                "purge",
                str(purge_feature_root),
                "binance",
                purge_window.start_utc,
                purge_window.end_utc,
                "a" * 64,
                "b" * 64,
            ),
            FeatureRootSpec(
                "validation",
                str(validation_feature_root),
                "binance",
                validation_window.start_utc,
                validation_window.end_utc,
                "c" * 64,
                "d" * 64,
            ),
        )
    )
    resolver = AlphaMaxFundingBoundaryResolver(lookup, admitted)
    timestamp_ms = int(datetime(2025, 6, 8, tzinfo=UTC).timestamp() * 1000)
    data_dict = dict.fromkeys(admitted, ((timestamp_ms, 100.0, 101.0, 99.0, 100.0, 10.0),))

    activation = construct_alpha_max_engine(
        preflight,
        output_root=str(root),
        phase="validation_train_fit",
        manifest_path=materialized.path,
        admitted_symbols=admitted,
        phase_id="validation",
        nominal_cost_bps=30,
        raw_root=str((tmp_path / "raw-validation").resolve()),
        ordered_lookup=lookup,
        funding_resolver=resolver,
        data_dict=data_dict,
    )

    validate_alpha_max_engine_activation(activation)
    assert type(activation.backtest) is Backtest
    assert type(activation.backtest.data_handler) is HistoricParquetWindowedDataHandler
    assert type(activation.backtest.strategy) is ArtifactPortfolioModeStrategy
    assert type(activation.backtest.portfolio) is Portfolio
    assert type(activation.backtest.execution_handler) is SimulatedExecutionHandler
    assert activation.backtest.symbol_list is admitted
    assert activation.backtest.data_handler.symbol_list is admitted
    assert activation.backtest.portfolio.symbol_list is admitted
    assert set(activation.constructor_plan.portfolio_kwargs) == {
        "fill_application_attribution_sink",
        "funding_boundary_resolver",
        "full_event_equity_sink",
        "reporting_sampling_timeframe",
    }
    fill_sink = activation.backtest.portfolio.fill_application_attribution_sink
    equity_sink = activation.backtest.portfolio.full_event_equity_sink
    assert fill_sink.__self__ is activation.attribution_collector
    assert fill_sink.__func__ is AlphaMaxAttributionCollector.record_application
    assert equity_sink.__self__ is activation.full_event_equity_tracker
    assert activation.backtest.portfolio.reporting_sampling_timeframe == "4h"
    assert activation.backtest.execution_handler.pricing_trace_evidence == ()

    # A crossed reporting boundary is priced from the completed native 4h
    # bucket even when sparse raw data has no row at the exact boundary and a
    # later row carries a hostile close.  The inclusive final endpoint settles
    # funding exactly once before it is emitted.
    reporting_start = datetime(2025, 6, 8, tzinfo=UTC)
    reporting_end = reporting_start + timedelta(hours=8)
    completed_close = 101.0
    hostile_later_close = 999.0
    aggregator = alpha_max_runner.TimeframeAggregator(timeframes=["4h"])
    for symbol in admitted:
        aggregator.update_from_1s_batch(
            symbol,
            (
                (
                    int((reporting_start + timedelta(seconds=5)).timestamp() * 1000),
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    1.0,
                ),
                (
                    int(
                        (reporting_start + timedelta(hours=3, minutes=59, seconds=55)).timestamp()
                        * 1000
                    ),
                    completed_close,
                    completed_close,
                    completed_close,
                    completed_close,
                    1.0,
                ),
                (
                    int((reporting_start + timedelta(hours=4, seconds=5)).timestamp() * 1000),
                    hostile_later_close,
                    hostile_later_close,
                    hostile_later_close,
                    hostile_later_close,
                    1.0,
                ),
            ),
        )
    activation.backtest.timeframe_aggregator = aggregator
    activation.backtest.portfolio.current_holdings["cash"] = 1_000.0
    activation.backtest.portfolio.current_positions[admitted[0]] = 2.0
    funding_boundaries: list[datetime] = []

    def record_funding(_portfolio: Portfolio, boundary: datetime) -> None:
        funding_boundaries.append(boundary)

    monkeypatch.setattr(Portfolio, "_apply_funding", record_funding)
    fanout = alpha_max_runner._AlphaMaxFoldEquityFanout(
        AlphaMaxStreamingEquityTracker(),
        aggregate_scale=1.0,
        reporting_start=reporting_start,
        reporting_end=reporting_end,
    )
    fanout.bind_backtest(activation.backtest)
    fanout.observe(
        (
            (reporting_start + timedelta(hours=4, seconds=5)).timestamp(),
            12_345.0,
        )
    )
    fanout.settle_day_end(reporting_end, settle_funding=True)

    endpoints = fanout.reporting_endpoints
    assert endpoints[0].equity == pytest.approx(1_000.0 + (2.0 * completed_close))
    assert endpoints[0].equity != pytest.approx(1_000.0 + (2.0 * hostile_later_close))
    assert funding_boundaries == [reporting_end]
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="reporting_boundary_sequence_invalid",
    ):
        fanout.settle_day_end(reporting_end, settle_funding=True)
    assert funding_boundaries == [reporting_end]


def test_prelock_and_historical_bundles_are_exclusive_immutable_and_duplicate_safe(
    tmp_path: Path,
) -> None:
    prelock_root = (tmp_path / "prelock").resolve()
    stable = {"status/matrix.json": b'{"status":"complete"}\n'}
    prelock = create_alpha_max_prelock_bundle(
        str(prelock_root),
        stable,
        prelock_champion=None,
        selected_candidate_id=None,
    )
    before = {
        path.relative_to(prelock_root): path.read_bytes()
        for path in prelock_root.rglob("*")
        if path.is_file()
    }

    assert Path(prelock.seal_path).is_file()
    assert prelock_root.stat().st_mode & 0o222 == 0
    with pytest.raises(AlphaMaxRuntimeContractError, match="output_root_exists"):
        create_alpha_max_prelock_bundle(
            str(prelock_root),
            stable,
            prelock_champion=None,
            selected_candidate_id=None,
        )

    historical = create_alpha_max_historical_package(
        str(prelock_root),
        str((tmp_path / "historical-1").resolve()),
        {"report.json": b'{"report_only":true}\n'},
        completion_id="evaluation-001",
    )
    after = {
        path.relative_to(prelock_root): path.read_bytes()
        for path in prelock_root.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert Path(historical.seal_path).is_file()
    with pytest.raises(AlphaMaxRuntimeContractError, match="completion_duplicate"):
        create_alpha_max_historical_package(
            str(prelock_root),
            str((tmp_path / "historical-2").resolve()),
            {"report.json": b'{"report_only":true}\n'},
            completion_id="evaluation-001",
        )
