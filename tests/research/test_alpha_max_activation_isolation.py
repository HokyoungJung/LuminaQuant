from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

import lumina_quant.research.alpha_max_engine_runner as runner
import lumina_quant.strategies.artifact_portfolio_mode as artifact_mode

from lumina_quant.backtesting.backtest import Backtest, FastQueue
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.research.alpha_max_engine_runner import (
    AlphaMaxRuntimeContractError,
    construct_alpha_max_engine,
    preflight_alpha_max_runtime_contract,
    validate_alpha_max_engine_activation,
)
from lumina_quant.research.alpha_max_evidence import (
    AlphaMaxFundingBoundaryResolver,
    AlphaMaxOrderedFundingLookup,
    FeatureRootSpec,
    materialize_alpha_max_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_SOURCE = (
    REPO_ROOT / "configs/research/alpha_max_portfolio_20260711_listing_aware.json"
).resolve()
INCUMBENT_AUDIT_SOURCE = (REPO_ROOT / ".omx/plans/alpha-max-incumbent-resolution-v1.json").resolve()


@pytest.fixture(autouse=True)
def _clean_lq_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("LQ_"):
            monkeypatch.delenv(key, raising=False)


@dataclass(slots=True)
class _ActivationHarness:
    preflight: runner.AlphaMaxRuntimePreflight
    admitted: tuple[str, ...]
    output_root: Path
    manifest_phase: str
    manifest_path: Path
    phase_id: str
    raw_root: Path
    lookup: AlphaMaxOrderedFundingLookup
    resolver: AlphaMaxFundingBoundaryResolver
    data_dict: dict[str, tuple[tuple[object, ...], ...]]
    config_path: Path

    def construct(self):
        return construct_alpha_max_engine(
            self.preflight,
            output_root=str(self.output_root),
            phase=self.manifest_phase,
            manifest_path=str(self.manifest_path),
            admitted_symbols=self.admitted,
            phase_id=self.phase_id,
            nominal_cost_bps=30,
            raw_root=str(self.raw_root),
            ordered_lookup=self.lookup,
            funding_resolver=self.resolver,
            data_dict=self.data_dict,
        )


def _copy_config(tmp_path: Path) -> Path:
    path = (tmp_path / "alpha-max-config.json").resolve()
    path.write_bytes(CONFIG_SOURCE.read_bytes())
    return path


def _root_specs(
    preflight: runner.AlphaMaxRuntimePreflight,
    tmp_path: Path,
    root_ids: tuple[str, ...],
) -> tuple[FeatureRootSpec, ...]:
    specs: list[FeatureRootSpec] = []
    for index, root_id in enumerate(root_ids):
        path = (tmp_path / "features" / root_id).resolve()
        path.mkdir(parents=True, exist_ok=True)
        window = preflight.phase_windows[root_id]
        specs.append(
            FeatureRootSpec(
                root_id,
                str(path),
                "binance",
                window.start_utc,
                window.end_utc,
                f"{index + 1:064x}",
                f"{index + 11:064x}",
            )
        )
    return tuple(specs)


def _build_harness(
    tmp_path: Path,
    *,
    phase_id: str = "validation",
    manifest_phase: str | None = None,
) -> _ActivationHarness:
    config_path = _copy_config(tmp_path)
    preflight = preflight_alpha_max_runtime_contract(str(config_path))
    admitted = preflight.candidate_symbols[:5]
    phase = manifest_phase or (
        "prelock_final_refit" if phase_id.startswith("historical_20") else "validation_train_fit"
    )
    output_root = (tmp_path / "run").resolve()
    (output_root / "manifests/validation_train_fit").mkdir(parents=True)
    (output_root / "manifests/prelock_final_refit").mkdir()
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    node = next(
        row
        for row in config_payload["current_trial_registry"]["nodes"]
        if row["row_id"] == "component_trend_1x"
    )
    materialized = materialize_alpha_max_manifest(
        node,
        {"component_trend_1x": 1.0},
        1.0,
        phase,
        str(config_path),
        str(output_root),
        preflight.candidate_symbols,
        admitted,
        "d" * 64,
    )
    expected_roots = runner._alpha_max_expected_root_sequence(phase_id)
    lookup = AlphaMaxOrderedFundingLookup(_root_specs(preflight, tmp_path, expected_roots))
    resolver = AlphaMaxFundingBoundaryResolver(lookup, admitted)
    window = preflight.phase_windows[phase_id]
    timestamp_ms = int(
        datetime.fromisoformat(window.start_utc.replace("Z", "+00:00")).timestamp() * 1000
    )
    data_dict = dict.fromkeys(
        admitted,
        ((timestamp_ms, 100.0, 101.0, 99.0, 100.0, 10.0),),
    )
    raw_root = (tmp_path / "raw").resolve()
    raw_root.mkdir()
    return _ActivationHarness(
        preflight=preflight,
        admitted=admitted,
        output_root=output_root,
        manifest_phase=phase,
        manifest_path=Path(materialized.path),
        phase_id=phase_id,
        raw_root=raw_root,
        lookup=lookup,
        resolver=resolver,
        data_dict=data_dict,
        config_path=config_path,
    )


def _event_probe(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    observed: list[str] = []
    original_put = FastQueue.put

    def recording_put(self, item, block=True, timeout=None):
        observed.append(str(getattr(item, "type", type(item).__name__)).upper())
        return original_put(self, item, block=block, timeout=timeout)

    monkeypatch.setattr(FastQueue, "put", recording_put)
    return observed


def _assert_zero_economic_events(observed: list[str], resolver) -> None:
    assert observed == []
    assert resolver.ledger == ()


@pytest.mark.parametrize(
    "phase_id",
    (
        "warmup",
        "train",
        "purge",
        "validation",
        "embargo",
        "historical_2025_09_partial",
    ),
)
def test_all_phase_activations_bind_strict_lookup_resolver_and_raw_accessor_before_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase_id: str,
) -> None:
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path, phase_id=phase_id)

    activation = harness.construct()

    validate_alpha_max_engine_activation(activation)
    assert activation.constructor_plan.strict_data_handler_construction is True
    assert activation.backtest.strict_data_handler_construction is True
    assert activation.backtest.data_handler_kwargs["feature_lookup"] is harness.lookup
    assert activation.backtest.data_handler._feature_lookup is harness.lookup
    assert activation.funding_resolver is harness.resolver
    assert harness.resolver.ordered_lookup is harness.lookup
    assert harness.resolver.admitted_symbols is harness.admitted
    assert harness.resolver._bound_raw_accessor_owner is activation.backtest.data_handler
    assert activation.backtest.portfolio.bars is activation.backtest.data_handler
    _assert_zero_economic_events(observed, harness.resolver)


def _atomic_descriptor_swap(
    target: Path,
    replacement: Path,
    read: Any,
):
    retained = target.with_name(f".{target.name}.retained-a")
    os.replace(target, retained)
    os.replace(replacement, target)
    try:
        return read()
    finally:
        os.replace(target, replacement)
        os.replace(retained, target)


def test_transient_consumer_manifest_and_config_b_are_rejected_after_valid_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path)
    manifest_a = json.loads(harness.manifest_path.read_text(encoding="utf-8"))
    config_b_payload = json.loads(harness.config_path.read_text(encoding="utf-8"))
    config_b_payload["incumbent_resolution"]["rows"][0]["resolution_reason"] = (
        "hostile transient audit B"
    )
    config_b_bytes = runner._canonical_bytes(config_b_payload) + b"\n"
    config_b = harness.config_path.with_name("config-b.json")
    config_b.write_bytes(config_b_bytes)

    manifest_b_payload = json.loads(json.dumps(manifest_a))
    manifest_b_payload["candidate_symbols"] = list(
        reversed(manifest_b_payload["candidate_symbols"])
    )
    manifest_b_payload["children"][0]["candidate_symbols"] = list(
        reversed(manifest_b_payload["children"][0]["candidate_symbols"])
    )
    manifest_b_payload["children"][0]["netting_group_gross_cap"] = 1.5
    manifest_b_payload["source_artifacts"][0]["sha256"] = hashlib.sha256(config_b_bytes).hexdigest()
    manifest_b = harness.manifest_path.with_name("manifest-b.json")
    manifest_b.write_bytes(runner._canonical_bytes(manifest_b_payload) + b"\n")

    original_json = artifact_mode.read_artifact_json
    original_bytes = artifact_mode.read_artifact_bytes
    original_resolve = artifact_mode.resolve_portfolio_mode_definition
    consumed_receipts: list[tuple[str, ...]] = []

    def transient_manifest(path, *, artifact_id):
        assert artifact_id == "artifact_portfolio_manifest"
        return _atomic_descriptor_swap(
            Path(path),
            manifest_b,
            lambda: original_json(path, artifact_id=artifact_id),
        )

    def transient_config(path, *, artifact_id):
        assert artifact_id == "source:alpha_max_config"
        return _atomic_descriptor_swap(
            Path(path),
            config_b,
            lambda: original_bytes(path, artifact_id=artifact_id),
        )

    def capture_valid_definition(token: str):
        definition = original_resolve(token)
        assert "manifest_fail_closed_reason" not in definition.source_artifacts
        consumed_receipts.append(
            tuple(receipt.sha256 for receipt in definition.artifact_read_receipts)
        )
        return definition

    monkeypatch.setattr(artifact_mode, "read_artifact_json", transient_manifest)
    monkeypatch.setattr(artifact_mode, "read_artifact_bytes", transient_config)
    monkeypatch.setattr(
        artifact_mode, "resolve_portfolio_mode_definition", capture_valid_definition
    )

    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="portfolio_manifest_activation_mismatch",
    ):
        harness.construct()

    assert consumed_receipts == [
        (
            hashlib.sha256(runner._canonical_bytes(manifest_b_payload) + b"\n").hexdigest(),
            hashlib.sha256(config_b_bytes).hexdigest(),
        )
    ]
    assert harness.manifest_path.read_bytes() == runner._canonical_bytes(manifest_a) + b"\n"
    assert harness.config_path.read_bytes() == CONFIG_SOURCE.read_bytes()
    _assert_zero_economic_events(observed, harness.resolver)


@pytest.mark.parametrize("mutation", ("same_bytes_rewrite", "atomic_identical_replace"))
def test_consumer_open_time_identity_mutations_fail_closed_before_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path)
    original = artifact_mode.read_artifact_json
    invoked = False

    def hostile_read(path, *, artifact_id):
        nonlocal invoked
        invoked = True
        target = Path(path)
        payload = target.read_bytes()
        if mutation == "same_bytes_rewrite":
            target.write_bytes(payload)
        else:
            replacement = target.with_name("atomic-identical.json")
            replacement.write_bytes(payload)
            os.replace(replacement, target)
        return original(path, artifact_id=artifact_id)

    monkeypatch.setattr(artifact_mode, "read_artifact_json", hostile_read)

    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="portfolio_manifest_activation_mismatch",
    ):
        harness.construct()

    assert invoked is True
    _assert_zero_economic_events(observed, harness.resolver)


@pytest.mark.parametrize("mutation", ("hardlink", "target_symlink", "ancestor_symlink"))
def test_manifest_path_hostility_rejects_before_engine_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path)
    if mutation == "hardlink":
        os.link(harness.manifest_path, harness.manifest_path.with_name("extra-link.json"))
    elif mutation == "target_symlink":
        retained = harness.manifest_path.with_name("retained.json")
        harness.manifest_path.rename(retained)
        harness.manifest_path.symlink_to(retained)
    else:
        phase_dir = harness.manifest_path.parent
        retained = phase_dir.with_name(f"{phase_dir.name}-retained")
        phase_dir.rename(retained)
        phase_dir.symlink_to(retained, target_is_directory=True)

    backtest_calls = 0

    def forbidden_backtest(*args, **kwargs):
        nonlocal backtest_calls
        backtest_calls += 1
        raise AssertionError("engine construction must not start")

    monkeypatch.setattr(runner, "Backtest", forbidden_backtest)
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="portfolio_manifest_activation_mismatch",
    ):
        harness.construct()

    assert backtest_calls == 0
    _assert_zero_economic_events(observed, harness.resolver)


@pytest.mark.parametrize(
    "mutation",
    (
        "receipt_missing",
        "receipt_reordered",
        "receipt_wrong_id",
        "receipt_extra",
        "source_mismatch",
        "definition_mismatch",
    ),
)
def test_consumer_receipt_and_definition_mutations_fail_closed_before_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path)
    original_resolve = artifact_mode.resolve_portfolio_mode_definition
    original_copy = artifact_mode._apply_component_param_overrides

    if mutation == "receipt_missing":

        def lossy_copy(definition, overrides):
            copied = original_copy(definition, overrides)
            return replace(copied, artifact_read_receipts=())

        monkeypatch.setattr(artifact_mode, "_apply_component_param_overrides", lossy_copy)
    else:

        def hostile_resolve(token: str):
            definition = original_resolve(token)
            if mutation == "receipt_reordered":
                return replace(
                    definition,
                    artifact_read_receipts=tuple(reversed(definition.artifact_read_receipts)),
                )
            if mutation == "receipt_wrong_id":
                manifest_receipt = replace(
                    definition.artifact_read_receipts[0],
                    artifact_id="wrong-manifest-id",
                )
                return replace(
                    definition,
                    artifact_read_receipts=(
                        manifest_receipt,
                        definition.artifact_read_receipts[1],
                    ),
                )
            if mutation == "receipt_extra":
                return replace(
                    definition,
                    artifact_read_receipts=(
                        *definition.artifact_read_receipts,
                        definition.artifact_read_receipts[-1],
                    ),
                )
            if mutation == "source_mismatch":
                sources = dict(definition.source_artifacts)
                sources["manifest_source_artifact:alpha_max_config"] = str(
                    harness.config_path.with_name("other.json")
                )
                return replace(definition, source_artifacts=sources)
            component = replace(definition.components[0], weight=0.5)
            return replace(definition, components=(component,))

        monkeypatch.setattr(artifact_mode, "resolve_portfolio_mode_definition", hostile_resolve)

    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="portfolio_manifest_activation_mismatch",
    ):
        harness.construct()

    _assert_zero_economic_events(observed, harness.resolver)


def test_stale_consumer_source_fails_closed_before_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path)

    class FarFutureDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(2200, 1, 1, tzinfo=UTC)
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(artifact_mode, "datetime", FarFutureDateTime)
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="portfolio_manifest_activation_mismatch",
    ):
        harness.construct()

    _assert_zero_economic_events(observed, harness.resolver)


def test_historical_activation_cannot_reuse_validation_manifest_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path, phase_id="validation")
    harness.manifest_phase = "prelock_final_refit"
    harness.phase_id = "historical_2025_09_partial"
    harness.lookup = AlphaMaxOrderedFundingLookup(
        _root_specs(
            harness.preflight,
            tmp_path / "historical",
            ("embargo", "historical_exposed_evaluation"),
        )
    )
    harness.resolver = AlphaMaxFundingBoundaryResolver(harness.lookup, harness.admitted)

    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="portfolio_manifest_activation_mismatch",
    ):
        harness.construct()

    _assert_zero_economic_events(observed, harness.resolver)


def test_alpha_strict_handler_rejection_is_never_retried_without_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path)
    calls: list[dict[str, object]] = []

    class RejectingHandler:
        def __init__(self, *args, **kwargs):
            calls.append(dict(kwargs))
            raise TypeError("hostile handler rejects alpha kwargs")

    monkeypatch.setattr(runner, "HistoricParquetWindowedDataHandler", RejectingHandler)

    with pytest.raises(TypeError, match="hostile handler rejects alpha kwargs"):
        harness.construct()

    assert len(calls) == 1
    assert set(calls[0]) == {
        "backtest_poll_seconds",
        "backtest_window_seconds",
        "feature_db_path",
        "feature_exchange",
        "feature_lookup",
        "market_window_parity_v2_enabled",
    }
    assert calls[0]["feature_lookup"] is harness.lookup
    _assert_zero_economic_events(observed, harness.resolver)


@pytest.mark.parametrize(
    "mutation",
    (
        "backtest_constructor",
        "portfolio_constructor",
        "config_receipt",
        "incumbent_audit_bytes",
    ),
)
def test_preflight_descriptor_constructor_and_audit_copies_cannot_reach_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path)
    if mutation == "backtest_constructor":
        harness.preflight = replace(
            harness.preflight,
            backtest_constructor=MappingProxyType({}),
        )
    elif mutation == "portfolio_constructor":
        harness.preflight = replace(
            harness.preflight,
            portfolio_strategy_constructor=MappingProxyType({}),
        )
    elif mutation == "config_receipt":
        harness.preflight = replace(
            harness.preflight,
            config_receipt=replace(harness.preflight.config_receipt, sha256="0" * 64),
        )
    else:
        harness.preflight = replace(
            harness.preflight,
            incumbent_resolution_bytes=b"{}",
        )
    backtest_calls = 0

    def forbidden_backtest(*args, **kwargs):
        nonlocal backtest_calls
        backtest_calls += 1
        raise AssertionError("engine construction must not start")

    monkeypatch.setattr(runner, "Backtest", forbidden_backtest)
    with pytest.raises(AlphaMaxRuntimeContractError, match="alpha_max_runtime_preflight_invalid"):
        harness.construct()

    assert backtest_calls == 0
    _assert_zero_economic_events(observed, harness.resolver)


@pytest.mark.parametrize("mutation", ("missing_predecessor", "later_pair"))
def test_wrong_ordered_root_contract_rejects_before_engine_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path)
    if mutation == "missing_predecessor":
        object.__setattr__(harness.lookup, "_root_specs", (harness.lookup.root_specs[-1],))
        invalid_lookup = harness.lookup
    else:
        invalid_lookup = AlphaMaxOrderedFundingLookup(
            _root_specs(harness.preflight, tmp_path / "later", ("validation", "embargo"))
        )
    harness.lookup = invalid_lookup
    harness.resolver = AlphaMaxFundingBoundaryResolver(invalid_lookup, harness.admitted)
    backtest_calls = 0

    def forbidden_backtest(*args, **kwargs):
        nonlocal backtest_calls
        backtest_calls += 1
        raise AssertionError("engine construction must not start")

    monkeypatch.setattr(runner, "Backtest", forbidden_backtest)
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="alpha_max_feature_root_sequence_mismatch",
    ):
        harness.construct()

    assert backtest_calls == 0
    _assert_zero_economic_events(observed, harness.resolver)


@pytest.mark.parametrize(
    "mutation",
    (
        "handler_lookup_copy",
        "ambient_feature_path",
        "activation_lookup_copy",
        "lookup_root_sequence",
        "activation_resolver_copy",
        "resolver_lookup_copy",
        "resolver_admitted_copy",
        "bound_accessor_owner",
        "raw_accessor_function",
        "portfolio_bars",
        "missing_resolver",
        "extra_portfolio_kwarg",
        "plan_strict_false",
    ),
)
def test_post_construction_lookup_resolver_and_accessor_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path)
    activation = harness.construct()
    copied_lookup = AlphaMaxOrderedFundingLookup(
        _root_specs(harness.preflight, tmp_path / "copy", ("purge", "validation"))
    )
    if mutation == "handler_lookup_copy":
        activation.backtest.data_handler._feature_lookup = copied_lookup
    elif mutation == "ambient_feature_path":
        activation.backtest.data_handler_kwargs["feature_db_path"] = "/tmp/ambient-feature-root"
    elif mutation == "activation_lookup_copy":
        object.__setattr__(activation, "ordered_lookup", copied_lookup)
    elif mutation == "lookup_root_sequence":
        object.__setattr__(
            activation.ordered_lookup, "_root_specs", (harness.lookup.root_specs[-1],)
        )
    elif mutation == "activation_resolver_copy":
        object.__setattr__(
            activation,
            "funding_resolver",
            AlphaMaxFundingBoundaryResolver(harness.lookup, harness.admitted),
        )
    elif mutation == "resolver_lookup_copy":
        object.__setattr__(activation.funding_resolver, "_ordered_lookup", copied_lookup)
    elif mutation == "resolver_admitted_copy":
        object.__setattr__(
            activation.funding_resolver,
            "_admitted_symbols",
            (*harness.admitted,),
        )
    elif mutation == "bound_accessor_owner":
        object.__setattr__(activation.funding_resolver, "_bound_raw_accessor_owner", object())
    elif mutation == "raw_accessor_function":
        activation.backtest.data_handler.get_latest_raw_point = lambda *args, **kwargs: None
    elif mutation == "portfolio_bars":
        activation.backtest.portfolio.bars = object()
    elif mutation == "missing_resolver":
        object.__setattr__(activation.backtest.portfolio, "_funding_boundary_resolver", None)
    elif mutation == "extra_portfolio_kwarg":
        activation.backtest.portfolio_kwargs["ambient_resolver"] = object()
    else:
        object.__setattr__(
            activation,
            "constructor_plan",
            replace(activation.constructor_plan, strict_data_handler_construction=False),
        )

    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="portfolio_manifest_activation_mismatch",
    ):
        validate_alpha_max_engine_activation(activation)

    _assert_zero_economic_events(observed, activation.funding_resolver)


def test_activated_resolver_rejects_unadmitted_candidate_before_lookup_or_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path)
    activation = harness.construct()
    lookup_calls = 0
    original_lookup = AlphaMaxOrderedFundingLookup.get_latest_point

    def recording_lookup(self, *args, **kwargs):
        nonlocal lookup_calls
        lookup_calls += 1
        return original_lookup(self, *args, **kwargs)

    monkeypatch.setattr(AlphaMaxOrderedFundingLookup, "get_latest_point", recording_lookup)
    outside = next(
        symbol for symbol in harness.preflight.candidate_symbols if symbol not in harness.admitted
    )
    with pytest.raises(ValueError, match="outside_admitted_domain"):
        activation.funding_resolver.resolve(
            symbol=outside,
            boundary_ms=1_749_340_800_000,
            qty=1.0,
            latest_datetime=datetime(2025, 6, 8, tzinfo=UTC),
            raw_point_accessor=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("raw accessor must not run")
            ),
        )

    assert lookup_calls == 0
    assert activation.funding_resolver._bound_raw_accessor_owner is activation.backtest.data_handler
    _assert_zero_economic_events(observed, activation.funding_resolver)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("path", "hostile/report-latest.json"),
        ("git_blob_oid", "0" * 40),
        ("content_sha256", "0" * 64),
        ("resolution_status", "resolved"),
        ("resolution_reason", "hostile reason"),
        ("audit_sha256", "0" * 64),
    ),
)
def test_incumbent_audit_mutations_reject_before_row_or_event_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    observed = _event_probe(monkeypatch)
    config_path = _copy_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if field == "audit_sha256":
        payload["normative_sources"]["incumbent_resolution_audit_sha256"] = value
        expected = "alpha_max_incumbent_resolution_audit_hash_mismatch"
    elif field in {"path", "git_blob_oid", "content_sha256"}:
        payload["incumbent_resolution"]["rows"][0]["frozen_audit_files"][0][field] = value
        expected = "alpha_max_incumbent_resolution_mismatch"
    else:
        payload["incumbent_resolution"]["rows"][0][field] = value
        expected = "alpha_max_incumbent_resolution_mismatch"
    config_path.write_bytes(runner._canonical_bytes(payload) + b"\n")
    backtest_calls = 0

    def forbidden_backtest(*args, **kwargs):
        nonlocal backtest_calls
        backtest_calls += 1
        raise AssertionError("row construction must not start")

    monkeypatch.setattr(runner, "Backtest", forbidden_backtest)
    with pytest.raises(AlphaMaxRuntimeContractError, match=expected):
        preflight_alpha_max_runtime_contract(str(config_path))

    assert backtest_calls == 0
    assert observed == []


def test_embedded_incumbent_audit_is_used_without_runtime_omx_or_glob_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planning_value = json.loads(INCUMBENT_AUDIT_SOURCE.read_text(encoding="utf-8"))
    expected_bytes = runner._canonical_bytes(planning_value)
    forbidden_audit_paths = tuple(
        str((REPO_ROOT / file["path"]).resolve())
        for row in planning_value["rows"]
        for file in row["frozen_audit_files"]
    )
    original_open = os.open

    def guarded_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None:
            requested = os.path.abspath(os.fspath(path))
            if ".omx" in Path(requested).parts:
                raise AssertionError("runtime attempted to open .omx")
            if requested in forbidden_audit_paths:
                raise AssertionError("runtime attempted to reopen an incumbent audit source")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def forbidden_glob(*args, **kwargs):
        raise AssertionError("runtime filesystem discovery is forbidden")

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(Path, "glob", forbidden_glob)
    monkeypatch.setattr(Path, "rglob", forbidden_glob)
    observed = _event_probe(monkeypatch)
    harness = _build_harness(tmp_path)

    activation = harness.construct()

    assert harness.preflight.incumbent_resolution_bytes == expected_bytes
    assert (
        harness.preflight.incumbent_resolution_audit_sha256
        == runner.ALPHA_MAX_INCUMBENT_RESOLUTION_AUDIT_SHA256
    )
    validate_alpha_max_engine_activation(activation)
    _assert_zero_economic_events(observed, activation.funding_resolver)


def test_generic_legacy_manifest_keeps_sorted_multi_source_receipt_cardinality(
    tmp_path: Path,
) -> None:
    source_alpha = (tmp_path / "alpha.json").resolve()
    source_zeta = (tmp_path / "zeta.json").resolve()
    source_alpha.write_text('{"ready":true,"value":"alpha"}\n', encoding="utf-8")
    source_zeta.write_text('{"ready":true,"value":"zeta"}\n', encoding="utf-8")
    provenance = {
        "selection_inputs": ["train", "validation"],
        "uses_current_fold_oos": False,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_objective": False,
    }
    correlation = {
        **provenance,
        "ready": True,
        "source": "train_validation_correlation_matrix",
        "uses_locked_oos_for_correlation": False,
    }
    manifest = {
        "gross_cap": 1.0,
        "cash_weight": 0.25,
        "optimizer_provenance": provenance,
        "correlation_input_provenance": correlation,
        "source_artifacts": [
            {
                "id": "zeta",
                "path": str(source_zeta),
                "sha256": hashlib.sha256(source_zeta.read_bytes()).hexdigest(),
                "max_age_hours": 876000,
                "ready": True,
                "portfolio_ready": True,
            },
            {
                "id": "alpha",
                "path": str(source_alpha),
                "sha256": hashlib.sha256(source_alpha.read_bytes()).hexdigest(),
                "max_age_hours": 876000,
                "ready": True,
                "portfolio_ready": True,
            },
        ],
        "children": [
            {
                "candidate_id": "leaf-a",
                "name": "Leaf A",
                "strategy_class": "MovingAverageCrossStrategy",
                "symbols": ["BTC/USDT"],
                "params": {"short_window": 4, "long_window": 12},
                "weight": 0.75,
                "leaf_gross": 0.75,
                "leaf_gross_cap": 1.0,
                "netting_group": "btc",
                "netting_group_gross_cap": 1.0,
                "source_artifact_id": "zeta",
                "ready": True,
                "portfolio_ready": True,
                "no_current_fold_oos_provenance": True,
                "train_validation_optimizer_provenance": True,
                "uses_current_fold_oos": False,
                "uses_locked_oos_for_selection": False,
                "uses_locked_oos_for_correlation": False,
                "optimizer_provenance": provenance,
                "correlation_input_provenance": correlation,
            }
        ],
    }
    manifest_path = (tmp_path / "legacy-manifest.json").resolve()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    definition = artifact_mode.resolve_portfolio_mode_definition(f"manifest:{manifest_path}")

    assert definition.components
    assert tuple(receipt.artifact_id for receipt in definition.artifact_read_receipts) == (
        "artifact_portfolio_manifest",
        "source:alpha",
        "source:zeta",
    )


def test_legacy_default_fallbacks_and_optional_resolver_remain_green() -> None:
    handler_calls: list[dict[str, object]] = []

    class LegacyHandler:
        def __init__(self, events, csv_dir, symbols, start, end, data_dict, **kwargs):
            handler_calls.append(dict(kwargs))
            if kwargs:
                raise TypeError("legacy handler rejects kwargs")
            self.symbol_list = symbols
            self.continue_backtest = False

    class LegacyStrategy:
        def __init__(self, bars, events):
            self.bars = bars
            self.events = events
            self.decision_cadence_seconds = 60

    class LegacyPortfolio:
        def __init__(self, bars, events, start, config, **kwargs):
            self.bars = bars
            self.kwargs = kwargs

    class LegacyExecution:
        def __init__(self, events, bars, config):
            self.bars = bars

    config = SimpleNamespace(
        TIMEFRAME="1s",
        DECISION_CADENCE_SECONDS=60,
        SKIP_AHEAD_ENABLED=False,
    )
    legacy = Backtest(
        csv_dir="/tmp/legacy",
        symbol_list=["BTCUSDT"],
        start_date=datetime(2025, 1, 1, tzinfo=UTC),
        end_date=datetime(2025, 1, 2, tzinfo=UTC),
        data_handler_cls=LegacyHandler,
        execution_handler_cls=LegacyExecution,
        portfolio_cls=LegacyPortfolio,
        strategy_cls=LegacyStrategy,
        data_handler_kwargs={"legacy_optional": True},
        config=config,
    )
    assert handler_calls == [{"legacy_optional": True}, {}]
    assert "funding_boundary_resolver" not in legacy.portfolio.kwargs

    actual_portfolio = Portfolio(
        SimpleNamespace(symbol_list=["BTCUSDT"]),
        FastQueue(),
        datetime(2025, 1, 1, tzinfo=UTC),
        SimpleNamespace(INITIAL_CAPITAL=1_000.0),
        record_history=False,
        track_metrics=False,
        record_trades=False,
    )
    assert actual_portfolio.funding_boundary_resolver is None
    assert (
        artifact_mode.resolve_portfolio_mode_definition("risk_off_mode").artifact_read_receipts
        == ()
    )
