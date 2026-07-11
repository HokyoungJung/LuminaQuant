from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

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
CONFIG_PATH = REPO_ROOT / "configs/research/alpha_max_portfolio_20260710.json"
CONTRACT_PATH = REPO_ROOT / "configs/research/alpha_max_contract_manifest_20260710.json"


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


def test_cli_parsers_expose_only_the_two_frozen_surfaces() -> None:
    prelock = _load("alpha_max_prelock_cli", PRELOCK_PATH)
    historical = _load("alpha_max_historical_cli", HISTORICAL_PATH)

    assert _option_strings(prelock.build_parser()) == {
        "--config",
        "--contract-manifest",
        "--exchange",
        "--output-root",
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
    assert _option_strings(historical.build_parser()) == {
        "--sealed-prelock-directory",
        "--embargo-feature-root",
        "--historical-evaluation-raw-root",
        "--historical-evaluation-feature-root",
        "--exchange",
        "--output-root",
    }


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


def test_clean_cli_main_delegates_exact_parsed_namespace(monkeypatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("LQ_"):
            monkeypatch.delenv(key, raising=False)
    module = _load("alpha_max_prelock_delegate", PRELOCK_PATH)
    seen = {}
    monkeypatch.setattr(module, "_execute", lambda args: seen.update(vars(args)) or 0)
    argv = []
    for option in (
        "config",
        "contract-manifest",
        "output-root",
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
    assert len(seen) == 14


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
            exchange="binance",
            output_root=str(output),
            **roots,
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".prelock-output.staging-*"))


def _complete_validation_matrix_bytes() -> bytes:
    nodes = json.loads(CONFIG_PATH.read_text())["current_trial_registry"]["nodes"]
    statuses = []
    for row in sorted(nodes, key=lambda value: value["row_id"]):
        row_id = row["row_id"]
        resolvable = not row_id.startswith("incumbent_") and not row_id.startswith("diagnostic_")
        for nominal in ALPHA_MAX_COST_CELL_BPS:
            statuses.append(
                {
                    "engine_constructed": resolvable,
                    "nominal_cost_bps": nominal,
                    "row_id": row_id,
                    "status": (
                        "resolved_engine_cell_complete"
                        if resolvable
                        else "incumbent_replay_unavailable"
                        if row_id.startswith("incumbent_")
                        else "diagnostic_report_only"
                    ),
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
        )

    assert not output.exists()
