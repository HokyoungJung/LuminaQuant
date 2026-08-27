from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lumina_quant.research.alpha_max_engine_runner import (
    ALPHA_MAX_COST_CELL_BPS,
    AlphaMaxRuntimeContractError,
    AmbientLQEnvironmentError,
    create_alpha_max_prelock_bundle,
    run_alpha_max_historical_process,
    run_alpha_max_prelock_process,
)


@pytest.fixture(autouse=True)
def _clear_ambient_lq_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Process-boundary tests start hermetic unless a case injects poison."""
    for key in tuple(os.environ):
        if key.startswith("LQ_"):
            monkeypatch.delenv(key, raising=False)


REPO_ROOT = Path(__file__).resolve().parents[2]
PRELOCK_PATH = REPO_ROOT / "scripts/research/run_alpha_max_prelock.py"
HISTORICAL_PATH = REPO_ROOT / "scripts/research/run_alpha_max_historical_evaluation.py"
PROCESS_FIXTURE_PATH = REPO_ROOT / "tests/research/_alpha_max_cli_process_fixture.py"
CONFIG_PATH = REPO_ROOT / "configs/research/alpha_max_portfolio_20260711_listing_aware.json"
CONTRACT_PATH = (
    REPO_ROOT / "configs/research/alpha_max_contract_manifest_20260711_listing_aware.json"
)
PRELOCK_OPTIONS = {
    "--config",
    "--contract-manifest",
    "--prior-trial-blob",
    "--exchange",
    "--output-root",
    "--checkpoint-root",
    "--warmup-raw-root",
    "--warmup-feature-root",
    "--train-raw-root",
    "--train-feature-root",
    "--purge-raw-root",
    "--purge-feature-root",
    "--validation-raw-root",
    "--validation-feature-root",
    "--embargo-raw-root",
    "--embargo-feature-root",
}
HISTORICAL_OPTIONS = {
    "--sealed-prelock-directory",
    "--embargo-feature-root",
    "--historical-evaluation-raw-root",
    "--historical-evaluation-feature-root",
    "--exchange",
    "--output-root",
    "--checkpoint-root",
}
BASELINE_LQ_KEYS = tuple(sorted(key for key in os.environ if key.startswith("LQ_")))

# Frozen section-8 coverage ledger. ``covered`` means the named test uses a real
# child process and the production boundary for the asserted slice. ``partial``
# names narrower instrumentation evidence without claiming the entire P-item.
# ``blocked`` records the fixture/seam needed instead of manufacturing a passing
# substitute. The ledger itself is tested below so every P01-P26 remains visible.
PROCESS_SPEC_COVERAGE: dict[str, tuple[str, str | None, str]] = {
    "P01": ("covered", "test_p01_prelock_help_process_exposes_no_historical_surface", "help"),
    "P02": ("covered", "test_p02_prelock_process_rejects_nonfrozen_arguments", "argv"),
    "P03": (
        "covered",
        "test_p03_public_prelock_is_byte_stable_under_every_historical_tree_poison",
        "public child plus production prelock bundle",
    ),
    "P04": (
        "covered",
        "test_p04_public_process_keeps_prelock_immutable_and_refuses_overwrite",
        "public child plus production lifecycle",
    ),
    "P05": (
        "covered",
        "test_p05_public_historical_process_rejects_every_inner_binding_mismatch",
        "outer-seal-preserving hostile bindings before replay",
    ),
    "P06": (
        "covered",
        "test_p06_public_prelock_materializes_inventory_bound_boundary_capsules",
        "production capsule materialization and readback",
    ),
    "P07": (
        "covered",
        "test_p07_public_historical_process_accepts_only_post_boundary_roots",
        "production root sealing and chronology gate",
    ),
    "P08": (
        "covered",
        "test_p08_public_prelock_seals_the_complete_frozen_matrix",
        "production matrix control and immutable inventory",
    ),
    "P09": (
        "covered",
        "test_p09_public_historical_process_is_one_touch_and_append_only",
        "production completion claim and immutable package",
    ),
    "P10": (
        "covered",
        "test_p10_historical_value_poison_cannot_change_prelock_or_selection_identity",
        "actual historical root reseal and production snapshot",
    ),
    "P11": (
        "covered",
        "test_p11_production_matrix_control_executes_each_allowed_physical_schedule_once",
        "production row-cost-fold control with deterministic replay-data stub; not physical market replay",
    ),
    "P12": (
        "covered",
        "test_p12_public_all_reject_run_has_no_promotion_and_complete_ledger",
        "production selection and terminal",
    ),
    "P13": (
        "covered",
        "test_p13_public_historical_leader_disagreement_is_report_only",
        "production historical ranking and terminal",
    ),
    "P14": (
        "covered",
        "test_p14_public_process_enforces_frozen_chronology_and_final_endpoint",
        "production preflight/parser/root sealing",
    ),
    "P15": (
        "covered",
        "test_p15_actual_process_artifacts_use_only_exposed_provenance_language",
        "help plus sealed output language",
    ),
    "P16": (
        "covered",
        "test_p16_public_terminal_collision_precedence_is_singular",
        "three production terminal scenarios",
    ),
    "P17": (
        "covered",
        "test_p17_public_process_rejects_every_owned_root_isolation_attack",
        "production root and contract sealing",
    ),
    "P18": (
        "covered",
        "test_p18_cli_process_rejects_every_lq_environment_key_before_help",
        "environment",
    ),
    "P19": (
        "covered",
        "test_p19_public_prelock_uses_only_the_frozen_git_trial_inventory",
        "same-target poison comparison and sealed ledger",
    ),
    "P20": (
        "covered",
        "test_p20_every_selectable_pair_uses_actual_constructor_kwargs_and_sink_validation",
        "68 actual Backtest constructors plus hostile doubles",
    ),
    "P21": (
        "covered",
        "test_p21_registry_candidate_and_admission_identities_are_isolated",
        "production registry/manifests plus hostile preflight/admission",
    ),
    "P22": ("covered", "test_p22_historical_process_surface_and_forbidden_arguments", "help/argv"),
    "P23": (
        "covered",
        "test_p23_descriptor_open_manifest_and_config_swaps_fail_before_events",
        "actual consumer descriptor opens after a valid production seal",
    ),
    "P24": (
        "covered",
        "test_p24_lookup_identity_mutations_fail_actual_activation_before_events",
        "actual activation lookup identities and root sequence",
    ),
    "P25": (
        "covered",
        "test_p25_resolver_and_raw_accessor_mutations_fail_before_events",
        "actual resolver/accessor activation identities",
    ),
    "P26": (
        "covered",
        "test_p26_public_prelock_rejects_embedded_incumbent_audit_mutations",
        "public child plus production incumbent-audit preflight",
    ),
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _option_strings(parser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option != "-h" and option != "--help"
    }


def _clean_subprocess_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("LQ_")}


def _run_cli_process(
    path: Path,
    *argv: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), *argv],
        cwd=REPO_ROOT,
        env=environment if environment is not None else _clean_subprocess_environment(),
        text=True,
        capture_output=True,
        check=False,
    )


def _help_option_strings(output: str) -> set[str]:
    return set(re.findall(r"--[a-z][a-z0-9-]*", output)) - {"--help"}


def _placeholder_prelock_argv() -> list[str]:
    argv = [
        "--config",
        "/not-opened/config.json",
        "--contract-manifest",
        "/not-opened/contract.json",
        "--prior-trial-blob",
        "/not-opened/prior-trials.json",
        "--exchange",
        "binance",
        "--output-root",
        "/not-opened/output",
        "--checkpoint-root",
        "/not-opened/checkpoints",
    ]
    for phase in ("warmup", "train", "purge", "validation", "embargo"):
        for kind in ("raw", "feature"):
            argv.extend((f"--{phase}-{kind}-root", f"/not-opened/{phase}-{kind}"))
    return argv


def _placeholder_historical_argv() -> list[str]:
    return [
        "--sealed-prelock-directory",
        "/not-opened/prelock",
        "--embargo-feature-root",
        "/not-opened/embargo-feature",
        "--historical-evaluation-raw-root",
        "/not-opened/historical-raw",
        "--historical-evaluation-feature-root",
        "/not-opened/historical-feature",
        "--exchange",
        "binance",
        "--output-root",
        "/not-opened/output",
        "--checkpoint-root",
        "/not-opened/checkpoints",
    ]


def test_physical_process_spec_coverage_ledger_enumerates_p01_through_p26() -> None:
    assert tuple(PROCESS_SPEC_COVERAGE) == tuple(f"P{index:02d}" for index in range(1, 27))
    for status, test_name, rationale in PROCESS_SPEC_COVERAGE.values():
        assert status in {"covered", "partial", "blocked"}
        assert rationale
        if status == "blocked":
            assert test_name is None
        else:
            assert test_name is not None and callable(globals().get(test_name))


def test_cli_parsers_expose_only_the_two_frozen_surfaces() -> None:
    prelock = _load("alpha_max_prelock_cli", PRELOCK_PATH)
    historical = _load("alpha_max_historical_cli", HISTORICAL_PATH)

    assert _option_strings(prelock.build_parser()) == PRELOCK_OPTIONS
    assert _option_strings(historical.build_parser()) == HISTORICAL_OPTIONS


def test_p01_prelock_help_process_exposes_no_historical_surface() -> None:
    result = _run_cli_process(PRELOCK_PATH, "--help")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert _help_option_strings(result.stdout) == PRELOCK_OPTIONS
    assert "historical" not in result.stdout.lower()


@pytest.mark.parametrize(
    ("forbidden_option", "forbidden_value"),
    (
        ("--profile", "hostile"),
        ("--runtime-config", "/not-opened/runtime.json"),
        ("--historical-evaluation-raw-root", "/not-opened/historical-raw"),
        ("--historical-evaluation-feature-root", "/not-opened/historical-feature"),
        ("--prior-trial-inventory", "/not-opened/prior.json"),
    ),
)
def test_p02_prelock_process_rejects_nonfrozen_arguments(
    forbidden_option: str,
    forbidden_value: str,
) -> None:
    result = _run_cli_process(
        PRELOCK_PATH,
        *_placeholder_prelock_argv(),
        forbidden_option,
        forbidden_value,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert f"unrecognized arguments: {forbidden_option}" in result.stderr


@pytest.mark.parametrize(
    ("forbidden_option", "forbidden_value"),
    (
        ("--embargo-raw-root", "/not-opened/embargo-raw"),
        ("--profile", "hostile"),
        ("--runtime-config", "/not-opened/runtime.json"),
        ("--validation-feature-root", "/not-opened/validation-feature"),
    ),
)
def test_p22_historical_process_surface_and_forbidden_arguments(
    forbidden_option: str,
    forbidden_value: str,
) -> None:
    help_result = _run_cli_process(HISTORICAL_PATH, "--help")
    assert help_result.returncode == 0, help_result.stderr
    assert help_result.stderr == ""
    assert _help_option_strings(help_result.stdout) == HISTORICAL_OPTIONS

    rejected = _run_cli_process(
        HISTORICAL_PATH,
        *_placeholder_historical_argv(),
        forbidden_option,
        forbidden_value,
    )
    assert rejected.returncode == 2
    assert rejected.stdout == ""
    assert f"unrecognized arguments: {forbidden_option}" in rejected.stderr


def test_p15_process_help_uses_only_exposed_provenance_language() -> None:
    forbidden = {"untouched", "locked", "prospective", "confirmatory"}
    for path in (PRELOCK_PATH, HISTORICAL_PATH):
        result = _run_cli_process(path, "--help")
        assert result.returncode == 0, result.stderr
        lowered = result.stdout.lower()
        assert not any(token in lowered for token in forbidden)
    historical = _run_cli_process(HISTORICAL_PATH, "--help")
    assert "exposed historical evaluation" in historical.stdout.lower()


@pytest.mark.parametrize(
    ("name", "path"),
    (("alpha_max_prelock_env", PRELOCK_PATH), ("alpha_max_historical_env", HISTORICAL_PATH)),
)
def test_cli_rejects_lq_before_parser_or_io(monkeypatch, name: str, path: Path) -> None:
    module = _load(name, path)
    monkeypatch.setenv("LQ_HOSTILE_OVERRIDE", "/must/not/open")
    monkeypatch.setattr(
        module,
        "build_parser",
        lambda: (_ for _ in ()).throw(AssertionError("parser constructed before environment gate")),
    )

    with pytest.raises(AmbientLQEnvironmentError, match="LQ_HOSTILE_OVERRIDE"):
        module.main([])


@pytest.mark.parametrize("path", (PRELOCK_PATH, HISTORICAL_PATH))
@pytest.mark.parametrize("environment_key", (*BASELINE_LQ_KEYS, "LQ_FOO"))
def test_p18_cli_process_rejects_every_lq_environment_key_before_help(
    path: Path,
    environment_key: str,
) -> None:
    environment = _clean_subprocess_environment()
    environment[environment_key] = "/must/not/open"

    result = _run_cli_process(path, "--help", environment=environment)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "usage:" not in result.stderr
    assert "AmbientLQEnvironmentError" in result.stderr
    assert f"ambient_lq_environment:{environment_key}" in result.stderr


def test_clean_cli_main_delegates_exact_parsed_namespace(monkeypatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("LQ_"):
            monkeypatch.delenv(key, raising=False)
    module = _load("alpha_max_prelock_delegate", PRELOCK_PATH)
    seen = {}
    monkeypatch.setattr(
        module,
        "_execute",
        lambda args, **kwargs: (
            seen.update(vars(args))
            or seen.update(bootstrap_inventory=kwargs["bootstrap_implementation_inventory"])
            or 0
        ),
    )
    argv = []
    for option in (
        "config",
        "contract-manifest",
        "prior-trial-blob",
        "output-root",
        "checkpoint-root",
        "warmup-raw-root",
        "warmup-feature-root",
        "train-raw-root",
        "train-feature-root",
        "purge-raw-root",
        "purge-feature-root",
        "validation-raw-root",
        "validation-feature-root",
        "embargo-raw-root",
        "embargo-feature-root",
    ):
        argv.extend((f"--{option}", f"/{option}"))
    argv.extend(("--exchange", "binance"))

    assert module.main(argv) == 0
    assert seen["exchange"] == "binance"
    assert seen["bootstrap_inventory"]
    assert len(seen) == 17


def test_historical_cli_forwards_checkpoint_root_exactly(monkeypatch) -> None:
    module = _load("alpha_max_historical_delegate", HISTORICAL_PATH)
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        module,
        "_execute",
        lambda args, **kwargs: (
            seen.update(vars(args))
            or seen.update(bootstrap_inventory=kwargs["bootstrap_implementation_inventory"])
            or 0
        ),
    )
    checkpoint_root = "/historical/checkpoint-root"
    assert (
        module.main(
            [
                "--sealed-prelock-directory",
                "/prelock",
                "--embargo-feature-root",
                "/embargo-feature",
                "--historical-evaluation-raw-root",
                "/historical-raw",
                "--historical-evaluation-feature-root",
                "/historical-feature",
                "--exchange",
                "binance",
                "--output-root",
                "/historical-output",
                "--checkpoint-root",
                checkpoint_root,
            ]
        )
        == 0
    )
    assert seen["checkpoint_root"] == checkpoint_root
    assert seen["bootstrap_inventory"]


def test_prelock_invalid_roots_leave_no_output_or_stage(tmp_path: Path) -> None:
    roots = {}
    for name in ("warmup", "train", "purge", "validation", "embargo"):
        for kind in ("raw", "feature"):
            path = (tmp_path / f"{name}-{kind}").resolve()
            path.mkdir()
            roots[f"{name}_{kind}_root"] = str(path)
    output = (tmp_path / "prelock-output").resolve()

    with pytest.raises(AlphaMaxRuntimeContractError, match="alpha_max_prelock_input_invalid"):
        run_alpha_max_prelock_process(
            config=str(CONFIG_PATH.resolve()),
            contract_manifest=str(CONTRACT_PATH.resolve()),
            prior_trial_blob=str(
                (
                    Path(__file__).resolve().parents[2]
                    / "var/reports/ultragoal_full_pool_strategy/"
                    "g004_frozen_candidate_manifest.json"
                ).resolve()
            ),
            exchange="binance",
            output_root=str(output),
            checkpoint_root=str((tmp_path / "prelock-checkpoints").resolve()),
            **roots,
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".prelock-output.staging-*"))


def test_prelock_cli_process_failure_leaves_no_output_or_stage(tmp_path: Path) -> None:
    roots: dict[str, Path] = {}
    for name in ("warmup", "train", "purge", "validation", "embargo"):
        for kind in ("raw", "feature"):
            path = (tmp_path / f"{name}-{kind}").resolve()
            path.mkdir()
            roots[f"{name}_{kind}_root"] = path
    output = (tmp_path / "prelock-output").resolve()
    argv = [
        "--config",
        str(CONFIG_PATH.resolve()),
        "--contract-manifest",
        str(CONTRACT_PATH.resolve()),
        "--prior-trial-blob",
        str(
            (
                Path(__file__).resolve().parents[2] / "var/reports/ultragoal_full_pool_strategy/"
                "g004_frozen_candidate_manifest.json"
            ).resolve()
        ),
        "--exchange",
        "binance",
        "--output-root",
        str(output),
        "--checkpoint-root",
        str((tmp_path / "prelock-checkpoints").resolve()),
    ]
    for name in ("warmup", "train", "purge", "validation", "embargo"):
        for kind in ("raw", "feature"):
            argv.extend(
                (
                    f"--{name}-{kind}-root",
                    str(roots[f"{name}_{kind}_root"]),
                )
            )

    result = _run_cli_process(PRELOCK_PATH, *argv)

    assert result.returncode != 0
    assert "AlphaMaxRuntimeContractError" in result.stderr
    assert "alpha_max_prelock_input_invalid" in result.stderr
    assert not output.exists()
    assert not tuple(tmp_path.glob(".prelock-output.staging-*"))


def _complete_validation_matrix_bytes() -> bytes:
    nodes = json.loads(CONFIG_PATH.read_text())["current_trial_registry"]["nodes"]
    statuses = []
    for row in sorted(nodes, key=lambda value: value["row_id"]):
        row_id = row["row_id"]
        for nominal in ALPHA_MAX_COST_CELL_BPS:
            common = {
                "nominal_cost_bps": nominal,
                "row_id": row_id,
            }
            if row_id.startswith("incumbent_"):
                statuses.append(
                    {
                        **common,
                        "capsule_sha256": None,
                        "engine_constructed": False,
                        "manifest_sha256": None,
                        "row_role": "incumbent_unavailable",
                        "selection_eligible": False,
                        "status": "incumbent_replay_unavailable",
                    }
                )
            elif row_id.startswith("diagnostic_"):
                statuses.append(
                    {
                        **common,
                        "capsule_sha256": None,
                        "engine_constructed": False,
                        "manifest_sha256": None,
                        "row_role": "track_b_diagnostic",
                        "selection_eligible": False,
                        "status": "diagnostic_report_only",
                    }
                )
            else:
                statuses.append(
                    {
                        **common,
                        "capsule_sha256": hashlib.sha256(
                            f"capsule:{row_id}:{nominal}".encode()
                        ).hexdigest(),
                        "cell_sha256": hashlib.sha256(
                            f"cell:{row_id}:{nominal}".encode()
                        ).hexdigest(),
                        "engine_constructed": True,
                        "manifest_sha256": hashlib.sha256(
                            f"manifest:{row_id}".encode()
                        ).hexdigest(),
                        "row_role": "resolvable_candidate",
                        "selection_eligible": True,
                        "status": "resolved_engine_cell_complete",
                    }
                )
    return (
        json.dumps(
            {
                "artifact_kind": "alpha_max_matrix_statuses.v1",
                "domain": "validation",
                "engine_cell_count": 68,
                "physical_fold_run_count": 816,
                "status_count": 84,
                "statuses": statuses,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def test_historical_invalid_roots_leave_no_output(tmp_path: Path) -> None:
    prelock = (tmp_path / "sealed-prelock").resolve()
    create_alpha_max_prelock_bundle(
        str(prelock),
        {
            "roots/feature/embargo.json": b"{}\n",
            "run/prelock_result.json": (
                b'{"engine_cell_count":68,"physical_fold_run_count":816,"prelock_champion":null}\n'
            ),
            "status/matrix.json": _complete_validation_matrix_bytes(),
        },
        prelock_champion=None,
        selected_candidate_id=None,
    )
    embargo = (tmp_path / "embargo-feature").resolve()
    historical_raw = (tmp_path / "historical-raw").resolve()
    historical_feature = (tmp_path / "historical-feature").resolve()
    for root in (embargo, historical_raw, historical_feature):
        root.mkdir()
    output = (tmp_path / "historical-output").resolve()

    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="alpha_max_historical_input_invalid",
    ):
        run_alpha_max_historical_process(
            sealed_prelock_directory=str(prelock),
            embargo_feature_root=str(embargo),
            historical_evaluation_raw_root=str(historical_raw),
            historical_evaluation_feature_root=str(historical_feature),
            exchange="binance",
            output_root=str(output),
            checkpoint_root=str((tmp_path / "historical-checkpoints").resolve()),
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".historical-output.staging-*"))


def test_p08_historical_process_accepts_exact_frozen_matrix_before_root_gate(
    tmp_path: Path,
) -> None:
    prelock = (tmp_path / "sealed-prelock").resolve()
    create_alpha_max_prelock_bundle(
        str(prelock),
        {
            "roots/feature/embargo.json": b"{}\n",
            "run/prelock_result.json": (
                b'{"engine_cell_count":68,"physical_fold_run_count":816,"prelock_champion":null}\n'
            ),
            "status/matrix.json": _complete_validation_matrix_bytes(),
        },
        prelock_champion=None,
        selected_candidate_id=None,
    )
    embargo = (tmp_path / "embargo-feature").resolve()
    historical_raw = (tmp_path / "historical-raw").resolve()
    historical_feature = (tmp_path / "historical-feature").resolve()
    for root in (embargo, historical_raw, historical_feature):
        root.mkdir()
    output = (tmp_path / "historical-output").resolve()

    result = _run_cli_process(
        HISTORICAL_PATH,
        "--sealed-prelock-directory",
        str(prelock),
        "--embargo-feature-root",
        str(embargo),
        "--historical-evaluation-raw-root",
        str(historical_raw),
        "--historical-evaluation-feature-root",
        str(historical_feature),
        "--exchange",
        "binance",
        "--output-root",
        str(output),
        "--checkpoint-root",
        str((tmp_path / "historical-checkpoints").resolve()),
    )

    assert result.returncode != 0
    assert "AlphaMaxRuntimeContractError" in result.stderr
    assert "alpha_max_historical_input_invalid" in result.stderr
    assert not output.exists()
    assert not tuple(tmp_path.glob(".historical-output.staging-*"))


def _run_physical_schedule_child(temp_root: Path) -> None:
    """Instrument expensive replay while retaining production matrix/process control flow."""
    import lumina_quant.research.alpha_max_engine_runner as runner
    import lumina_quant.research.alpha_max_evidence as evidence

    nodes = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["current_trial_registry"]["nodes"]
    calls: list[tuple[str, str, int, str]] = []

    def fold_inputs(domain: str) -> tuple[SimpleNamespace, ...]:
        return tuple(
            SimpleNamespace(fold_id=fold_id) for fold_id in runner._alpha_max_fold_ids(domain)
        )

    def prepared_rows(domain: str) -> dict[str, object]:
        return {
            row_id: runner._AlphaMaxPreparedReplayRow(
                manifest_receipt=SimpleNamespace(sha256=f"{index + 1:064x}"),
                fold_inputs=fold_inputs(domain),
                gross=1.0,
            )
            for index, row_id in enumerate(runner._ALPHA_MAX_RESOLVABLE_ROWS)
        }

    def replay_stub(
        _preflight,
        *,
        row_id: str,
        domain: str,
        nominal_cost_bps: int,
        fold_inputs: tuple[SimpleNamespace, ...],
        **_kwargs,
    ) -> SimpleNamespace:
        fold_runs = []
        for fold_input in fold_inputs:
            calls.append((domain, row_id, nominal_cost_bps, fold_input.fold_id))
            fold_runs.append(SimpleNamespace(split_or_fold_id=fold_input.fold_id))
        return SimpleNamespace(
            combined_primary_return_stream=None,
            status="complete",
            fold_runs=tuple(fold_runs),
        )

    def cell_stub(pre_gate, *, statistical_evidence=None) -> SimpleNamespace:
        assert statistical_evidence is None
        return SimpleNamespace(
            pre_gate_evidence=pre_gate,
            status="complete",
            selection_valid=True,
            capsule_receipts=(),
        )

    # Only the data/backtest replay and typed evidence packaging are stubbed. The
    # production row/cost/fold loops, cardinality checks, schedule validator, and
    # public historical process remain the code under test in this child process.
    runner._replay_alpha_max_cost_cell_pre_gate = replay_stub
    runner.build_alpha_max_cost_cell_evidence = cell_stub
    runner.AlphaMaxRowEvidence = lambda **values: SimpleNamespace(**values)
    runner.canonical_alpha_max_cost_cell_bytes = lambda _value: b"instrumented-cell\n"

    validation_matrix = runner._alpha_max_complete_domain_matrix(
        None,
        output_root=temp_root,
        phase="validation_train_fit",
        nodes=nodes,
        admitted_symbols=("BTCUSDT",),
        domain="validation",
        trial_ledger=None,
        prepared_rows=prepared_rows("validation"),
    )
    validation_calls = tuple(
        (row_id, nominal, fold_id)
        for domain, row_id, nominal, fold_id in calls
        if domain == "validation"
    )
    validation_expected = runner._alpha_max_physical_fold_schedule("validation")
    assert len(validation_calls) == len(set(validation_calls)) == len(validation_expected)
    assert set(validation_calls) == set(validation_expected)
    assert validation_matrix.physical_fold_run_count == 816

    prelock = (temp_root / "sealed-prelock").resolve()
    prelock.mkdir()
    marker = b"prelock-read-only\n"
    (prelock / "marker").write_bytes(marker)
    contract_manifest_bytes = CONTRACT_PATH.read_bytes()
    contract_manifest_path = prelock / "inputs/contract_manifest.json"
    contract_manifest_path.parent.mkdir()
    contract_manifest_path.write_bytes(contract_manifest_bytes)
    output = (temp_root / "historical-output").resolve()
    duplicate_output = (temp_root / "historical-output-duplicate").resolve()
    snapshot = SimpleNamespace(root_path=str(prelock))
    embargo_bytes = b'{"root":"embargo"}\n'

    def root_seal(
        *,
        root_id: str,
        root_kind: str,
        path: Path,
        payload: bytes,
        marker: str,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            availability_sha256=marker * 64,
            canonical_bytes=payload,
            content_sha256=marker * 64,
            inventory_sha256=marker * 64,
            path=str(path),
            root_id=root_id,
            root_kind=root_kind,
            sha256=marker * 64,
        )

    root_seals = {
        ("embargo", "feature"): SimpleNamespace(
            **vars(
                root_seal(
                    root_id="embargo",
                    root_kind="feature",
                    path=temp_root / "embargo-feature",
                    payload=embargo_bytes,
                    marker="a",
                )
            ),
        ),
        ("historical_exposed_evaluation", "raw"): SimpleNamespace(
            **vars(
                root_seal(
                    root_id="historical_exposed_evaluation",
                    root_kind="raw",
                    path=temp_root / "historical-raw",
                    payload=b"{}\n",
                    marker="b",
                )
            ),
        ),
        ("historical_exposed_evaluation", "feature"): SimpleNamespace(
            **vars(
                root_seal(
                    root_id="historical_exposed_evaluation",
                    root_kind="feature",
                    path=temp_root / "historical-feature",
                    payload=b"{}\n",
                    marker="c",
                )
            ),
        ),
    }
    seal_bytes = b'{"prelock_champion":null,"selected_candidate_id":null}\n'
    prelock_payload = b'{"physical_fold_run_count":816,"prelock_champion":null}\n'
    admission_sha256 = "a" * 64
    admission_computation_bytes = (
        json.dumps(
            {
                "admission_artifact_sha256": admission_sha256,
                "artifact_kind": "alpha_max_train_admission_computation.v1",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    liquidity_bucket_bytes = b'{"artifact_kind":"instrumented_train_liquidity_buckets"}\n'

    def read_artifact(_snapshot, relative: str) -> bytes:
        return {
            "roots/feature/embargo.json": embargo_bytes,
            "run/prelock_result.json": prelock_payload,
            "inputs/contract_manifest.json": contract_manifest_bytes,
            "inputs/prior_trial_inventory.json": b"{}\n",
            "admission/train.json": b"{}\n",
            "admission/train_computation.json": admission_computation_bytes,
            "admission/train_liquidity_buckets.json": liquidity_bucket_bytes,
            "selection/prelock.json": b"{}\n",
        }[relative]

    real_snapshot_bundle_tree = runner._snapshot_bundle_tree
    runner._snapshot_bundle_tree = lambda path, **kwargs: (
        snapshot if Path(path).resolve() == prelock else real_snapshot_bundle_tree(path, **kwargs)
    )
    runner._validate_prelock_snapshot = lambda _snapshot: ("1" * 64, seal_bytes)
    runner._validate_complete_alpha_max_prelock_matrix = lambda *_args, **_kwargs: None
    runner._alpha_max_root_validation = lambda *_args, **_kwargs: (root_seals, ())
    runner._validate_alpha_max_adjacent_feature_roots = lambda *_args, **_kwargs: None
    runner._read_alpha_max_prelock_artifact = read_artifact
    physical_preflight = runner.preflight_alpha_max_runtime_contract(CONFIG_PATH)
    runner.preflight_alpha_max_runtime_contract = lambda _path: physical_preflight
    runner._alpha_max_current_nodes = lambda _preflight: tuple(nodes)
    runner.build_alpha_max_trial_ledger = lambda *_args, **_kwargs: SimpleNamespace()
    evidence.validate_alpha_max_admission_artifact = lambda _raw: SimpleNamespace(
        admitted_symbols=("BTCUSDT",),
        sha256=admission_sha256,
    )
    runner.validate_alpha_max_train_liquidity_buckets = lambda _raw: SimpleNamespace(
        admitted_symbols=("BTCUSDT",),
        admission_computation_sha256=hashlib.sha256(admission_computation_bytes).hexdigest(),
        canonical_bytes=liquidity_bucket_bytes,
    )
    runner._alpha_max_trend_liquidity_falsifier_artifact = lambda *_args, **_kwargs: (
        b'{"artifact_kind":"instrumented_liquidity_falsifier"}\n'
    )
    runner._validate_admitted_symbols = lambda _preflight, symbols: tuple(symbols)
    runner._alpha_max_selection_from_bytes = lambda *_args, **_kwargs: SimpleNamespace(
        prelock_champion=None,
        selected_candidate_id=None,
    )
    retained_manifests: dict[str, object] = {}

    def retained_row(_snapshot, *, row_id: str):
        index = runner._ALPHA_MAX_RESOLVABLE_ROWS.index(row_id) + 1
        manifest = SimpleNamespace(
            row_id=row_id,
            phase="prelock_final_refit",
            path=f"/fixture/manifests/{row_id}.json",
            sha256=f"{index:064x}",
            byte_count=index,
        )
        retained_manifests[manifest.path] = manifest
        return (
            manifest,
            SimpleNamespace(),
            SimpleNamespace(),
            1.0,
        )

    runner._alpha_max_prelock_final_row_artifacts = retained_row
    runner.seal_alpha_max_manifest_activation = lambda *_args, manifest_path, **_kwargs: (
        SimpleNamespace(manifest_receipt=retained_manifests[str(manifest_path)])
    )
    runner._AlphaMaxBoundedRawLoader = lambda *_args, **_kwargs: SimpleNamespace()
    runner._alpha_max_build_fold_inputs = lambda *_args, domain, **_kwargs: fold_inputs(domain)
    runner._alpha_max_prepared_row_checkpoint_bytes = lambda prepared, *, domain: (
        runner._canonical_bytes(
            {
                "artifact_kind": "fixture_prepared_checkpoint.v1",
                "domain": domain,
                "gross_hex": prepared.gross.hex(),
                "row_id": prepared.manifest_receipt.row_id,
            }
        )
        + b"\n"
    )

    def restore_prepared_checkpoint(
        payload: bytes,
        *,
        manifest: object,
        domain: str,
        gross: float,
        **kwargs: object,
    ) -> object:
        assert runner._strict_json_object(payload) == {
            "artifact_kind": "fixture_prepared_checkpoint.v1",
            "domain": domain,
            "gross_hex": gross.hex(),
            "row_id": manifest.row_id,
        }
        if domain == "historical_exposed_evaluation":
            capsule_root = Path(kwargs["capsule_output_root"])
            for fold_id in runner._ALPHA_MAX_HISTORICAL_FOLD_IDS[1:]:
                relative = f"capsules/prelock_final_refit/{manifest.row_id}/{fold_id}.json"
                capsule_bytes = (
                    runner._canonical_bytes(
                        {
                            "artifact_kind": "fixture_historical_capsule.v1",
                            "fold_id": fold_id,
                            "row_id": manifest.row_id,
                        }
                    )
                    + b"\n"
                )
                path = capsule_root / relative
                if path.exists():
                    assert path.read_bytes() == capsule_bytes
                else:
                    runner._write_bundle_file_atomic(
                        capsule_root,
                        relative,
                        capsule_bytes,
                    )
        return runner._AlphaMaxPreparedReplayRow(
            manifest_receipt=manifest,
            fold_inputs=fold_inputs(domain),
            gross=gross,
        )

    runner._alpha_max_restore_prepared_row_checkpoint = restore_prepared_checkpoint
    runner.rank_alpha_max_historical_report = lambda _rows: SimpleNamespace(
        canonical_bytes=b'{"rows":[]}\n'
    )
    runner.build_alpha_max_terminal_state = lambda **_kwargs: SimpleNamespace(
        confirmation_status="not_applicable",
        historical_evaluation_leader=None,
        historical_exposure_status="complete_report_only",
        requires_fresh_confirmation=False,
        terminal_outcome="no_prelock_champion",
        to_payload=lambda: {"terminal_outcome": "no_prelock_champion"},
    )
    runner._alpha_max_root_artifacts = lambda _seals: {}
    runner._alpha_max_matrix_artifacts = lambda matrix: {
        "status/matrix.json": matrix.status_payload
    }

    class InstrumentedCellCheckpointStore:
        def __init__(
            self,
            _checkpoint_root: str,
            *,
            output_root: str,
            descriptor: object,
            config_bytes: bytes,
        ) -> None:
            assert descriptor
            assert config_bytes
            self.output_root = Path(output_root).resolve()
            self.display_output_root = self.output_root
            self._physical_schedule_sha256 = descriptor["physical_schedule_sha256"]
            self.descriptor_sha256 = "f" * 64
            self._precompute: dict[tuple[str, str], bytes] = {}

        def load(self, **_kwargs: object) -> None:
            return None

        def bind_output_root(self) -> Path:
            return self.output_root

        def seal(self, evidence_value: object, **_kwargs: object) -> object:
            return evidence_value

        def load_precompute(self, *, unit_kind: str, unit_id: str) -> bytes | None:
            return self._precompute.get((unit_kind, unit_id))

        def seal_precompute(
            self,
            *,
            unit_kind: str,
            unit_id: str,
            data_bytes: bytes,
        ) -> bytes:
            self._precompute[(unit_kind, unit_id)] = data_bytes
            return data_bytes

    runner._AlphaMaxCellCheckpointStore = InstrumentedCellCheckpointStore
    write_order: list[str] = []
    real_write_bundle_file = runner._write_bundle_file
    real_write_bundle_file_atomic = runner._write_bundle_file_atomic
    real_write_final_seal = runner._alpha_max_write_final_seal

    def recording_write_bundle_file(root: Path, relative_path: str, payload: bytes) -> Path:
        written = real_write_bundle_file(root, relative_path, payload)
        write_order.append(Path(relative_path).as_posix())
        return written

    runner._write_bundle_file = recording_write_bundle_file

    def recording_write_bundle_file_atomic(
        root: Path,
        relative_path: str,
        payload: bytes,
    ) -> Path:
        written = real_write_bundle_file_atomic(root, relative_path, payload)
        write_order.append(Path(relative_path).as_posix())
        return written

    runner._write_bundle_file_atomic = recording_write_bundle_file_atomic

    def recording_write_final_seal(fd: int, payload: bytes) -> None:
        real_write_final_seal(fd, payload)
        write_order.append("SEALED.json")

    runner._alpha_max_write_final_seal = recording_write_final_seal

    result = runner.run_alpha_max_historical_process(
        sealed_prelock_directory=str(prelock),
        embargo_feature_root=str(temp_root / "embargo-feature"),
        historical_evaluation_raw_root=str(temp_root / "historical-raw"),
        historical_evaluation_feature_root=str(temp_root / "historical-feature"),
        exchange="binance",
        output_root=str(output),
        checkpoint_root=str(temp_root / "historical-checkpoints"),
    )
    historical_calls = tuple(
        (row_id, nominal, fold_id)
        for domain, row_id, nominal, fold_id in calls
        if domain == "historical_exposed_evaluation"
    )
    historical_expected = runner._alpha_max_physical_fold_schedule("historical_exposed_evaluation")
    assert len(historical_calls) == len(set(historical_calls)) == len(historical_expected)
    assert set(historical_calls) == set(historical_expected)
    assert len(historical_calls) == 680
    assert result.exit_code == 0

    report = json.loads((output / "report/historical_result.json").read_bytes())
    assert report["prelock_champion"] is None
    assert report["physical_fold_run_count"] == 680
    seal = output / "SEALED.json"
    assert seal.is_file()
    assert write_order[-1] == "SEALED.json"
    seal_payload = json.loads(seal.read_bytes())
    inventory = seal_payload["historical_artifacts"]
    assert {entry["relative_path"] for entry in inventory} == {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path != seal
    }
    for entry in inventory:
        payload = (output / entry["relative_path"]).read_bytes()
        assert len(payload) == entry["byte_count"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o444 for path in output.rglob("*") if path.is_file()
    )
    assert (prelock / "marker").read_bytes() == marker

    overwrite_error = None
    try:
        runner.run_alpha_max_historical_process(
            sealed_prelock_directory=str(prelock),
            embargo_feature_root=str(temp_root / "embargo-feature"),
            historical_evaluation_raw_root=str(temp_root / "historical-raw"),
            historical_evaluation_feature_root=str(temp_root / "historical-feature"),
            exchange="binance",
            output_root=str(output),
            checkpoint_root=str(temp_root / "historical-checkpoints"),
        )
    except runner.AlphaMaxRuntimeContractError as exc:
        overwrite_error = str(exc)
    assert overwrite_error == "alpha_max_output_root_recovered_sealed"

    duplicate_error = None
    try:
        runner.run_alpha_max_historical_process(
            sealed_prelock_directory=str(prelock),
            embargo_feature_root=str(temp_root / "embargo-feature"),
            historical_evaluation_raw_root=str(temp_root / "historical-raw"),
            historical_evaluation_feature_root=str(temp_root / "historical-feature"),
            exchange="binance",
            output_root=str(duplicate_output),
            checkpoint_root=str(temp_root / "historical-checkpoints"),
        )
    except runner.AlphaMaxRuntimeContractError as exc:
        duplicate_error = str(exc)
    assert duplicate_error == "alpha_max_historical_completion_duplicate"
    assert not duplicate_output.exists()

    print(
        json.dumps(
            {
                "champion": None,
                "duplicate_error": duplicate_error,
                "historical_physical_runs": len(historical_calls),
                "overwrite_error": overwrite_error,
                "prelock_marker": marker.decode(),
                "sealed": seal.is_file(),
                "validation_physical_runs": len(validation_calls),
            },
            sort_keys=True,
        )
    )


@pytest.fixture(scope="module")
def physical_schedule_child_payload(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    tmp_path = tmp_path_factory.mktemp("alpha-max-physical-child")
    result = _run_cli_process(
        Path(__file__),
        "--alpha-max-physical-schedule-child",
        str(tmp_path.resolve()),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert type(payload) is dict
    return payload


def test_p04_child_process_preserves_prelock_and_refuses_overwrite(
    physical_schedule_child_payload: dict[str, object],
) -> None:
    assert physical_schedule_child_payload["prelock_marker"] == "prelock-read-only\n"
    assert (
        physical_schedule_child_payload["overwrite_error"]
        == "alpha_max_output_root_recovered_sealed"
    )


def test_p09_child_process_seals_once_and_refuses_duplicate_completion(
    physical_schedule_child_payload: dict[str, object],
) -> None:
    assert physical_schedule_child_payload["sealed"] is True
    assert (
        physical_schedule_child_payload["duplicate_error"]
        == "alpha_max_historical_completion_duplicate"
    )


def test_p10_child_process_keeps_prelock_bytes_stable(
    physical_schedule_child_payload: dict[str, object],
) -> None:
    assert physical_schedule_child_payload["prelock_marker"] == "prelock-read-only\n"
    assert physical_schedule_child_payload["champion"] is None


def test_p11_child_process_executes_exact_physical_schedules(
    physical_schedule_child_payload: dict[str, object],
) -> None:
    assert physical_schedule_child_payload["validation_physical_runs"] == 816
    assert physical_schedule_child_payload["historical_physical_runs"] == 680


@pytest.fixture(scope="module")
def public_process_payload(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    """Run all costly filesystem/public-process cases once in a fresh child."""
    tmp_path = tmp_path_factory.mktemp("alpha-max-public-process")
    result = _run_cli_process(PROCESS_FIXTURE_PATH, str(tmp_path.resolve()))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert type(payload) is dict
    return payload


def test_p03_public_prelock_is_byte_stable_under_every_historical_tree_poison(
    public_process_payload: dict[str, object],
) -> None:
    assert public_process_payload["p03"] == {
        "add": True,
        "chmod": True,
        "content": True,
        "remove": True,
        "rename": True,
        "touch": True,
    }


def test_p04_public_process_keeps_prelock_immutable_and_refuses_overwrite(
    public_process_payload: dict[str, object],
) -> None:
    p04 = public_process_payload["p04"]
    assert type(p04) is dict
    assert "alpha_max_output_root_recovered_sealed" in p04["overwrite_error"]
    assert p04["overwrite_before_replay"] is True
    assert p04["prelock_bytes_unchanged"] is True
    assert p04["prelock_modes_unchanged"] is True
    assert p04["historical_checkpoint_cells"] == 68
    assert p04["historical_checkpoint_folds"] == 680


def test_p05_public_historical_process_rejects_every_inner_binding_mismatch(
    public_process_payload: dict[str, object],
) -> None:
    p05 = public_process_payload["p05"]
    assert type(p05) is dict
    for binding in (
        "config",
        "runtime_contract",
        "source",
        "raw_feature",
        "capsule",
        "membership",
        "policy",
        "gross",
        "manifest",
    ):
        assert "Error:" in p05[binding]
        assert p05[f"{binding}_before_replay"] == "True"


def test_p06_public_prelock_materializes_inventory_bound_boundary_capsules(
    public_process_payload: dict[str, object],
) -> None:
    p06 = public_process_payload["p06"]
    assert p06 == {
        "all_before_historical": True,
        "all_inventory_bound": True,
        "capsule_count": 17,
        "inventory_independent": True,
    }


def test_p07_public_historical_process_accepts_only_post_boundary_roots(
    public_process_payload: dict[str, object],
) -> None:
    p07 = public_process_payload["p07"]
    for attack in (
        "historical_preboundary_raw",
        "historical_preboundary_feature",
        "historical_overlap_embargo",
    ):
        assert "alpha_max_historical_input_invalid" in p07[attack]
        assert p07[f"{attack}_before_replay"] == "True"


def test_p08_public_prelock_seals_the_complete_frozen_matrix(
    public_process_payload: dict[str, object],
) -> None:
    p08 = public_process_payload["p08"]
    assert p08["status_count"] == 84
    assert p08["engine_cell_count"] == 68
    assert p08["physical_fold_run_count"] == 816
    assert len(p08["row_ids"]) == 21
    assert p08["unavailable"] == [
        "incumbent_cross_asset_lead_lag_momentum",
        "incumbent_cross_candidate_hybrid_v3_5",
        "incumbent_track_a_dynamic_conviction_switch",
    ]
    assert p08["diagnostic"] == ["diagnostic_track_b_codex_lagged_leaf_router_grid"]
    assert p08["sealed_inventory_exact"] is True


def test_p09_public_historical_process_is_one_touch_and_append_only(
    public_process_payload: dict[str, object],
) -> None:
    p09 = public_process_payload["p09"]
    assert "alpha_max_output_root_recovered_sealed" in p09["overwrite_error"]
    assert "alpha_max_historical_completion_duplicate" in p09["duplicate_error"]
    assert p09["duplicate_output_absent"] is True
    assert p09["immutable"] is True
    assert p09["inventory_exact"] is True


def test_p10_historical_value_poison_cannot_change_prelock_or_selection_identity(
    public_process_payload: dict[str, object],
) -> None:
    p10 = public_process_payload["p10"]
    assert all(p10.values())


def test_p11_production_matrix_control_executes_each_allowed_physical_schedule_once(
    public_process_payload: dict[str, object],
) -> None:
    p11 = public_process_payload["p11"]
    for label, expected in (
        ("prelock-no-champion", 816),
        ("historical-no-champion", 680),
    ):
        observed = p11[label]
        assert observed["count"] == observed["unique"] == observed["expected"] == expected
        assert observed["forbidden_rows_absent"] is True


def test_p12_public_all_reject_run_has_no_promotion_and_complete_ledger(
    public_process_payload: dict[str, object],
) -> None:
    p12 = public_process_payload["p12"]
    assert p12["champion"] is None
    assert p12["selected"] is None
    assert p12["ranked"] == []
    assert p12["terminal"] == p12["terminal_artifact"] == "no_demonstrated_alpha"
    assert p12["num_trials"] == 1487
    assert p12["complete_matrix"] is True


def test_p13_public_historical_leader_disagreement_is_report_only(
    public_process_payload: dict[str, object],
) -> None:
    p13 = public_process_payload["p13"]
    assert p13["prelock_champion"] == "component_carry_1x"
    assert p13["historical_leader"] == "component_near_high_1x"
    assert p13["historical_leader"] != p13["prelock_champion"]
    assert p13["selected_candidate_id"] == p13["prelock_champion"]
    assert p13["report_selected_candidate_id"] == p13["prelock_champion"]
    assert p13["leader_differs"] is True
    assert p13["requires_fresh_confirmation"] is True


def test_p14_public_process_enforces_frozen_chronology_and_final_endpoint(
    public_process_payload: dict[str, object],
) -> None:
    p14 = public_process_payload["p14"]
    assert p14["chronology_exact"] is True
    assert set(p14["config_errors"]) == {
        "overlap",
        "regime",
        "shifted_boundary",
        "shortening",
    }
    assert all("Error:" in value for value in p14["config_errors"].values())
    assert p14["cli_override_code"] == 2
    assert "unrecognized arguments: --start-date" in p14["cli_override_message"]
    assert "alpha_max_historical_input_invalid" in p14["partial_endpoint_error"]
    assert p14["partial_endpoint_before_replay"] is True


def test_p15_actual_process_artifacts_use_only_exposed_provenance_language(
    public_process_payload: dict[str, object],
) -> None:
    p15 = public_process_payload["p15"]
    assert p15["forbidden_hits"] == []
    assert p15["historical_exposure_status"] == "committed_period_outcomes_observed"
    assert p15["requires_fresh_confirmation"] is True
    assert p15["report_status"] == "complete_report_only"


def test_p16_public_terminal_collision_precedence_is_singular(
    public_process_payload: dict[str, object],
) -> None:
    p16 = public_process_payload["p16"]
    assert p16["pass"]["terminal_outcome"] == ("prelock_champion_historical_robustness_passed")
    assert p16["fail"]["terminal_outcome"] == ("prelock_champion_historical_robustness_failed")
    assert p16["no_survivor"]["terminal_outcome"] == "no_demonstrated_alpha"
    for terminal in (p16["pass"], p16["fail"], p16["no_survivor"]):
        assert terminal["incumbent_comparison_status"] == "unavailable"
    assert p16["pass"]["leader_differs_from_prelock_champion"] is True
    assert p16["fail"]["leader_differs_from_prelock_champion"] is True
    assert p16["pass_report"]["selected_candidate_id"] == "component_carry_1x"
    assert p16["fail_report"]["selected_candidate_id"] == "component_carry_1x"


def test_p17_public_process_rejects_every_owned_root_isolation_attack(
    public_process_payload: dict[str, object],
) -> None:
    p17 = public_process_payload["p17"]
    for attack in (
        "gap",
        "duplicate_timestamps",
        "outside_interval",
        "nonadjacent_later_root",
        "purge_hidden_in_adjacent",
        "embargo_hash_mutation",
        "contract_manifest",
    ):
        assert "Error:" in p17[attack]
        assert p17[f"{attack}_before_replay"] == "True"
    assert p17["historical_poison_prelock_stable"] == "True"


def test_p19_public_prelock_uses_only_the_frozen_git_trial_inventory(
    public_process_payload: dict[str, object],
) -> None:
    p19 = public_process_payload["p19"]
    assert p19["prior_identical"] is True
    assert p19["ledger_identical"] is True
    assert p19["num_trials"] == 1487
    assert len(p19["prior_key_set_sha256"]) == 64
    assert len(p19["current_key_set_sha256"]) == 64


def test_p20_every_selectable_pair_uses_actual_constructor_kwargs_and_sink_validation(
    public_process_payload: dict[str, object],
) -> None:
    p20 = public_process_payload["p20"]
    assert p20["constructor_count"] == p20["constructor_pairs_unique"] == 68
    assert p20["forbidden_rows_absent"] is True
    assert p20["portfolio_kwargs"] == [
        "fill_application_attribution_sink",
        "full_event_equity_sink",
        "funding_boundary_resolver",
        "reporting_sampling_timeframe",
    ]
    assert p20["sink_identities"] is True
    assert p20["legacy_retry_calls"] == [{"legacy_optional": True}, {}]
    assert p20["legacy_alpha_kwargs_absent"] is True
    assert "SinkRejected:alpha_sink_kwargs_rejected" in p20["rejecting_error"]
    assert "portfolio_manifest_activation_mismatch" in p20["ignoring_error"]


def test_p21_registry_candidate_and_admission_identities_are_isolated(
    public_process_payload: dict[str, object],
) -> None:
    p21 = public_process_payload["p21"]
    assert len(p21["config_registry_sha256"]) == 64
    assert p21["all_nodes_ten_candidates"] is True
    assert len(p21["admitted_symbols"]) == 5
    assert p21["all_manifests_candidate_ten_active_admitted"] is True
    assert p21["cosmetic_prior_key_stable"] is True
    assert p21["symbol_reorder_key_stable"] is True
    assert p21["behavioral_prior_key_changed"] is True
    for attack in (
        "behavioral_node",
        "candidate_mapping",
        "alternate_source_id",
        "registry_reorder",
        "admitted_mapping",
    ):
        assert "Error:" in p21["errors"][attack]
        assert p21["errors"][f"{attack}_before_replay"] == "True"


def test_p23_descriptor_open_manifest_and_config_swaps_fail_before_events(
    public_process_payload: dict[str, object],
) -> None:
    p23 = public_process_payload["p23"]
    assert p23["targets_restored"] is True
    for attack in ("manifest_descriptor_swap", "config_descriptor_swap"):
        assert "portfolio_manifest_activation_mismatch" in p23[attack]
        assert p23[f"{attack}_before_events"] is True


def test_p24_lookup_identity_mutations_fail_actual_activation_before_events(
    public_process_payload: dict[str, object],
) -> None:
    p24 = public_process_payload["p24"]
    for attack in (
        "handler_lookup_copy",
        "activation_lookup_copy",
        "lookup_root_sequence",
    ):
        assert "portfolio_manifest_activation_mismatch" in p24[attack]
        assert p24[f"{attack}_before_events"] is True


def test_p25_resolver_and_raw_accessor_mutations_fail_before_events(
    public_process_payload: dict[str, object],
) -> None:
    p25 = public_process_payload["p25"]
    for attack in (
        "activation_resolver_copy",
        "resolver_lookup_copy",
        "resolver_admitted_copy",
        "bound_accessor_owner",
        "raw_accessor_function",
        "portfolio_bars",
        "missing_resolver",
    ):
        assert "portfolio_manifest_activation_mismatch" in p25[attack]
        assert p25[f"{attack}_before_events"] is True


def test_p26_public_prelock_rejects_embedded_incumbent_audit_mutations(
    public_process_payload: dict[str, object],
) -> None:
    p26 = public_process_payload["p26"]
    assert p26["embedded_audit_bytes_exact"] is True
    assert p26["embedded_audit_sha_exact"] is True
    for attack in (
        "audit_path",
        "audit_git_blob",
        "audit_content_sha",
        "resolution_status",
        "resolution_reason",
    ):
        assert "alpha_max_incumbent_resolution_mismatch" in p26[attack]
        assert p26[f"{attack}_before_replay"] is True
    assert "alpha_max_incumbent_resolution_audit_hash_mismatch" in p26["normative_audit_sha"]
    assert p26["normative_audit_sha_before_replay"] is True


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--alpha-max-physical-schedule-child":
        _run_physical_schedule_child(Path(sys.argv[2]))
    else:  # pragma: no cover - this module only exposes one child-process entry point
        raise SystemExit("alpha_max_cli_boundary_child_arguments_invalid")
