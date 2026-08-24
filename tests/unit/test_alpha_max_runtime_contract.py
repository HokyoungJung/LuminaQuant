from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import lumina_quant.configuration as configuration
from lumina_quant.portfolio.strategy_quality import StrategyQualityOverlay
from lumina_quant.research import alpha_max_engine_runner as runner
from lumina_quant.research.alpha_max_engine_runner import (
    ALPHA_MAX_CANDIDATE_SYMBOLS,
    ALPHA_MAX_CONFIG_FILE_SHA256,
    ALPHA_MAX_COST_CELL_BPS,
    ALPHA_MAX_RUNTIME_CONTRACT_SHA256,
    AlphaMaxBacktestConfig,
    AlphaMaxRuntimeContractError,
    AmbientLQEnvironmentError,
    FrozenRuntimeMutationError,
    UnfrozenRuntimeFieldError,
    alpha_max_common_rng_seed,
    alpha_max_common_rng_seed_payload,
    build_alpha_max_backtest_config,
    build_alpha_max_cost_cell_configs,
    build_alpha_max_engine_constructor_plan,
    preflight_alpha_max_runtime_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "configs/research/alpha_max_portfolio_20260711_listing_aware.json"
).resolve()
ADMITTED_SYMBOLS = ALPHA_MAX_CANDIDATE_SYMBOLS[:5]


@pytest.fixture(autouse=True)
def _remove_ambient_lq_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("LQ_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def preflight():
    return preflight_alpha_max_runtime_contract(CONFIG_PATH)


def _copy_config(tmp_path: Path) -> Path:
    target = (tmp_path / CONFIG_PATH.name).resolve()
    target.write_bytes(CONFIG_PATH.read_bytes())
    return target


def _mutated_config(tmp_path: Path, mutate) -> Path:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(payload)
    target = (tmp_path / "mutated.json").resolve()
    target.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def _poison(*args, **kwargs):
    del args, kwargs
    raise AssertionError("forbidden configuration/default/profile loader called")


def test_u40_descriptor_receipt_is_exact_and_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    original = runner.read_artifact_bytes

    def _record(path, *, artifact_id):
        calls.append((os.fspath(path), artifact_id))
        return original(path, artifact_id=artifact_id)

    monkeypatch.setattr(runner, "read_artifact_bytes", _record)

    result = preflight_alpha_max_runtime_contract(CONFIG_PATH)

    assert calls == [(str(CONFIG_PATH), "alpha_max_config")]
    assert result.config_receipt.requested_path == str(CONFIG_PATH)
    assert result.config_receipt.canonical_path == str(CONFIG_PATH)
    assert result.config_receipt.pre_fstat_identity == result.config_receipt.post_fstat_identity
    assert result.config_receipt.sha256 == ALPHA_MAX_CONFIG_FILE_SHA256
    assert result.runtime_contract_sha256 == ALPHA_MAX_RUNTIME_CONTRACT_SHA256
    assert hashlib.sha256(result.runtime_contract_bytes).hexdigest() == (
        ALPHA_MAX_RUNTIME_CONTRACT_SHA256
    )


def test_u40_ambient_lq_key_fails_before_descriptor_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def _forbidden_read(*args, **kwargs):
        nonlocal called
        del args, kwargs
        called = True
        raise AssertionError("descriptor read must not occur")

    monkeypatch.setattr(runner, "read_artifact_bytes", _forbidden_read)
    monkeypatch.setenv("LQ_PROFILE", "hostile")

    with pytest.raises(AmbientLQEnvironmentError, match="ambient_lq_environment:LQ_PROFILE"):
        preflight_alpha_max_runtime_contract(CONFIG_PATH)
    assert called is False


def test_u40_every_lq_prefix_is_rejected_and_relative_path_is_not_an_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("LQ_CONFIG_PATH", "LQ_PROFILE", "LQ_UNKNOWN_RUNTIME_FIELD"):
        monkeypatch.setenv(key, "poison")
        with pytest.raises(AmbientLQEnvironmentError, match="ambient_lq_environment"):
            preflight_alpha_max_runtime_contract(CONFIG_PATH)
        monkeypatch.delenv(key)

    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="alpha_max_config_path_not_explicit_canonical",
    ):
        preflight_alpha_max_runtime_contract(
            "configs/research/alpha_max_portfolio_20260711_listing_aware.json"
        )


def test_u40_yaml_profile_and_default_runtime_poison_are_never_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit_config = _copy_config(tmp_path)
    (tmp_path / "config.yaml").write_text("this: [is: invalid", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(configuration, "get_default_runtime_config", _poison)
    monkeypatch.setattr(configuration, "load_runtime_config", _poison)
    monkeypatch.setattr(configuration, "load_yaml_config", _poison)
    monkeypatch.setattr(configuration, "build_runtime_config", _poison)
    monkeypatch.setattr(Path, "read_text", _poison)

    result = preflight_alpha_max_runtime_contract(explicit_config)
    config = build_alpha_max_backtest_config(
        result,
        phase_id="validation_w01",
        admitted_symbols=ADMITTED_SYMBOLS,
        nominal_cost_bps=30,
    )

    assert config.TIMEFRAME == "1s"
    assert config.SYMBOLS is ADMITTED_SYMBOLS
    assert result.config_receipt.sha256 == ALPHA_MAX_CONFIG_FILE_SHA256


def test_u40_config_construction_rechecks_ambient_environment(preflight, monkeypatch) -> None:
    monkeypatch.setenv("LQ_CONFIG_PATH", "/tmp/late-poison.yaml")

    with pytest.raises(AmbientLQEnvironmentError, match="ambient_lq_environment"):
        build_alpha_max_backtest_config(
            preflight,
            phase_id="validation_w01",
            admitted_symbols=ADMITTED_SYMBOLS,
            nominal_cost_bps=20,
        )


def test_u41_allowlist_values_and_nested_runtime_state_are_immutable(preflight) -> None:
    config = build_alpha_max_backtest_config(
        preflight,
        phase_id="validation_w01",
        admitted_symbols=ADMITTED_SYMBOLS,
        nominal_cost_bps=20,
    )
    snapshot = config.runtime_attribute_snapshot()

    assert tuple(snapshot) == preflight.attribute_allowlist
    assert tuple(snapshot) == tuple(sorted(snapshot))
    assert set(snapshot) == set(preflight.static_attributes) | {
        "END_DATE",
        "RANDOM_SEED",
        "SLIPPAGE_RATE",
        "START_DATE",
        "SYMBOLS",
    }
    assert snapshot["TIMEFRAMES"] == ("1s", "4h", "1d")
    assert snapshot["SYMBOLS"] is ADMITTED_SYMBOLS
    assert snapshot["START_DATE"] == "2025-06-08T00:00:00Z"
    assert snapshot["END_DATE"] == "2025-06-15T00:00:00Z"
    assert snapshot["SLIPPAGE_RATE"] == 0.0015
    assert snapshot["SIM_LATENCY_MIN_BARS"] == snapshot["SIM_LATENCY_MAX_BARS"] == 1
    assert snapshot["DEFAULT_ORDER_TYPE"] == "MKT"
    assert snapshot["LIMIT_PRICE_MODE"] == "one_tick_worse"
    assert snapshot["LIMIT_TIME_IN_FORCE"] == "GTC"
    assert snapshot["SYMBOL_LIMITS"] == {
        symbol: {
            "min_notional": 5.0,
            "min_qty": 0.001,
            "price_tick_size": 1e-8,
            "qty_step": 0.001,
        }
        for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS
    }

    with pytest.raises(FrozenRuntimeMutationError, match="frozen_runtime_field:LEVERAGE"):
        config.LEVERAGE = 4
    with pytest.raises(FrozenRuntimeMutationError, match="frozen_runtime_field:LEVERAGE"):
        del config.LEVERAGE
    with pytest.raises(TypeError):
        snapshot["SYMBOL_LIMITS"]["BTCUSDT"]["min_qty"] = 1.0
    with pytest.raises(TypeError):
        snapshot["SYMBOL_LIMITS"]["BTCUSDT"] = {}


def test_u41_disabled_quality_empty_state_restores_without_unfrozen_read(preflight) -> None:
    config = build_alpha_max_backtest_config(
        preflight,
        phase_id="train",
        admitted_symbols=ADMITTED_SYMBOLS,
        nominal_cost_bps=10,
    )
    overlay = StrategyQualityOverlay(config)
    state = overlay.get_state()

    assert state["health"] == {}
    overlay.set_state(state)

    assert overlay.get_state() == state
    assert config.runtime_read_audit == ("STRATEGY_QUALITY_ENABLED",)


def test_u41_unknown_or_private_rt_read_fails_closed_and_audit_is_deterministic(
    preflight,
) -> None:
    config = build_alpha_max_backtest_config(
        preflight,
        phase_id="train",
        admitted_symbols=ADMITTED_SYMBOLS,
        nominal_cost_bps=10,
    )

    assert config.runtime_read_audit == ()
    assert config.TIMEFRAME == "1s"
    assert config.LEVERAGE == 3
    assert config.TIMEFRAME == "1s"
    expected_reads = ("TIMEFRAME", "LEVERAGE", "TIMEFRAME")
    assert config.runtime_read_audit == expected_reads
    expected_audit_bytes = json.dumps(
        list(expected_reads),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert config.runtime_read_audit_sha256 == hashlib.sha256(expected_audit_bytes).hexdigest()

    with pytest.raises(UnfrozenRuntimeFieldError, match="unfrozen_runtime_field:UNDECLARED"):
        getattr(config, "UNDECLARED", None)
    with pytest.raises(UnfrozenRuntimeFieldError, match="unfrozen_runtime_field:_rt"):
        getattr(config, "_rt", None)
    assert "_rt" not in AlphaMaxBacktestConfig.__slots__


def test_u41_direct_constructor_is_closed() -> None:
    with pytest.raises(AlphaMaxRuntimeContractError, match="alpha_max_config_constructor_private"):
        AlphaMaxBacktestConfig(
            attributes={},
            contract_sha256=ALPHA_MAX_RUNTIME_CONTRACT_SHA256,
            construction_token=object(),
        )


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_u41_nonfinite_json_is_rejected(tmp_path: Path, token: str) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        '"INITIAL_CAPITAL": 10000.0',
        f'"INITIAL_CAPITAL": {token}',
        1,
    )
    mutated = (tmp_path / "nonfinite.json").resolve()
    mutated.write_text(text, encoding="utf-8")

    with pytest.raises(AlphaMaxRuntimeContractError, match="nonfinite"):
        preflight_alpha_max_runtime_contract(mutated)


def test_u41_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        '"runtime_contract": {',
        '"runtime_contract": {},\n  "runtime_contract": {',
        1,
    )
    mutated = (tmp_path / "duplicate.json").resolve()
    mutated.write_text(text, encoding="utf-8")

    with pytest.raises(AlphaMaxRuntimeContractError, match="duplicate_json_key:runtime_contract"):
        preflight_alpha_max_runtime_contract(mutated)


def test_u49_exact_four_cost_cells_share_all_non_cost_runtime_bytes(preflight) -> None:
    configs = build_alpha_max_cost_cell_configs(
        preflight,
        phase_id="validation_w01",
        admitted_symbols=ADMITTED_SYMBOLS,
    )
    snapshots = [config.runtime_attribute_snapshot() for config in configs]

    assert tuple(cell.nominal_one_way_bps for cell in preflight.cost_cells) == (
        ALPHA_MAX_COST_CELL_BPS
    )
    assert tuple(cell.slippage_rate for cell in preflight.cost_cells) == (
        0.0005,
        0.001,
        0.0015,
        0.0025,
    )
    assert tuple(snapshot["SLIPPAGE_RATE"] for snapshot in snapshots) == (
        0.0005,
        0.001,
        0.0015,
        0.0025,
    )
    assert all(snapshot["SYMBOLS"] is ADMITTED_SYMBOLS for snapshot in snapshots)
    assert len({config.runtime_instance_sha256 for config in configs}) == 4

    for index, nominal_bps in enumerate(ALPHA_MAX_COST_CELL_BPS):
        assert snapshots[index]["RANDOM_SEED"] == alpha_max_common_rng_seed(
            "validation_w01", nominal_bps
        )
    excluded = {"RANDOM_SEED", "SLIPPAGE_RATE"}
    common = [
        {name: value for name, value in snapshot.items() if name not in excluded}
        for snapshot in snapshots
    ]
    assert common[1:] == [common[0], common[0], common[0]]


def test_u49_seed_schedule_is_exact_and_row_independent() -> None:
    expected_payload = b"alpha_max_20260710\0validation_w01\0" + b"20"
    assert alpha_max_common_rng_seed_payload("validation_w01", 20) == expected_payload
    expected = int.from_bytes(hashlib.sha256(expected_payload).digest()[:8], "big") % 2_147_483_647
    assert alpha_max_common_rng_seed("validation_w01", 20) == (expected or 1)


@pytest.mark.parametrize(
    ("field", "mutated_value"),
    [
        ("SIM_LATENCY_MIN_BARS", 0),
        ("SIM_LATENCY_MAX_BARS", 2),
        ("DEFAULT_ORDER_TYPE", "LMT"),
        ("LIMIT_PRICE_MODE", "mid"),
        ("LIMIT_TIME_IN_FORCE", "IOC"),
        ("LIMIT_PRICE_OFFSET_TICKS", 2),
    ],
)
def test_u49_order_and_latency_mutations_fail_the_sealed_contract(
    tmp_path: Path,
    field: str,
    mutated_value: object,
) -> None:
    mutated = _mutated_config(
        tmp_path,
        lambda payload: payload["runtime_contract"]["static_attributes"].__setitem__(
            field, mutated_value
        ),
    )

    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="alpha_max_runtime_contract_mismatch:static_attributes",
    ):
        preflight_alpha_max_runtime_contract(mutated)


def test_u49_cost_cell_mutation_fails_the_sealed_contract(tmp_path: Path) -> None:
    mutated = _mutated_config(
        tmp_path,
        lambda payload: payload["cost_cells"][0].__setitem__("slippage_rate", 0.0004),
    )

    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="alpha_max_runtime_contract_mismatch:cost_cells",
    ):
        preflight_alpha_max_runtime_contract(mutated)


def test_u50_constructor_plan_has_only_explicit_one_second_phase_owned_inputs(
    preflight,
) -> None:
    config = build_alpha_max_backtest_config(
        preflight,
        phase_id="validation_w01",
        admitted_symbols=ADMITTED_SYMBOLS,
        nominal_cost_bps=30,
    )
    feature_lookup = object()
    funding_resolver = object()
    attribution_sink = object()

    plan = build_alpha_max_engine_constructor_plan(
        preflight,
        config=config,
        feature_lookup=feature_lookup,
        funding_boundary_resolver=funding_resolver,
        fill_application_attribution_sink=attribution_sink,
    )

    assert plan.config is config
    assert plan.strategy_timeframe == "1s"
    assert plan.warmup_bars == 0
    assert plan.record_history is plan.track_metrics is plan.record_trades is True
    assert plan.strict_data_handler_construction is True
    assert dict(plan.data_handler_kwargs) == {
        "backtest_poll_seconds": 1,
        "backtest_window_seconds": 1,
        "feature_db_path": None,
        "feature_exchange": "binance",
        "feature_lookup": feature_lookup,
        "market_window_parity_v2_enabled": True,
    }
    assert plan.data_handler_kwargs["feature_lookup"] is feature_lookup
    assert plan.portfolio_kwargs["funding_boundary_resolver"] is funding_resolver
    assert plan.portfolio_kwargs["fill_application_attribution_sink"] is attribution_sink
    assert dict(plan.execution_handler_kwargs) == {"record_cost_attribution": True}
    assert plan.as_kwargs()["config"] is config
    with pytest.raises(TypeError):
        plan.data_handler_kwargs["backtest_poll_seconds"] = 2


def test_u50_default_runtime_poison_does_not_affect_config_or_plan(
    preflight,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(configuration, "get_default_runtime_config", _poison)
    monkeypatch.setattr(configuration, "load_runtime_config", _poison)
    monkeypatch.setattr(configuration, "load_yaml_config", _poison)
    config = build_alpha_max_backtest_config(
        preflight,
        phase_id="train",
        admitted_symbols=ADMITTED_SYMBOLS,
        nominal_cost_bps=10,
    )

    plan = build_alpha_max_engine_constructor_plan(
        preflight,
        config=config,
        feature_lookup=object(),
        funding_boundary_resolver=object(),
        fill_application_attribution_sink=object(),
    )

    assert config.BACKTEST_POLL_SECONDS == config.BACKTEST_WINDOW_SECONDS == 1
    assert config.BACKTEST_DECISION_SECONDS == config.DECISION_CADENCE_SECONDS == 1
    assert config.MARKET_WINDOW_PARITY_V2_ENABLED is True
    assert config.SKIP_AHEAD_ENABLED is False
    assert plan.data_handler_kwargs["feature_exchange"] == "binance"


@pytest.mark.parametrize(
    "missing",
    ["feature_lookup", "funding_boundary_resolver", "fill_application_attribution_sink"],
)
def test_u50_omitted_phase_owned_constructor_identity_is_rejected(preflight, missing: str) -> None:
    config = build_alpha_max_backtest_config(
        preflight,
        phase_id="train",
        admitted_symbols=ADMITTED_SYMBOLS,
        nominal_cost_bps=10,
    )
    kwargs = {
        "config": config,
        "feature_lookup": object(),
        "funding_boundary_resolver": object(),
        "fill_application_attribution_sink": object(),
    }
    kwargs[missing] = None

    with pytest.raises(AlphaMaxRuntimeContractError, match="required"):
        build_alpha_max_engine_constructor_plan(preflight, **kwargs)


def test_u50_late_ambient_environment_blocks_constructor_plan(
    preflight,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_alpha_max_backtest_config(
        preflight,
        phase_id="train",
        admitted_symbols=ADMITTED_SYMBOLS,
        nominal_cost_bps=10,
    )
    monkeypatch.setenv("LQ_ANYTHING", "late-poison")

    with pytest.raises(AmbientLQEnvironmentError, match="ambient_lq_environment"):
        build_alpha_max_engine_constructor_plan(
            preflight,
            config=config,
            feature_lookup=object(),
            funding_boundary_resolver=object(),
            fill_application_attribution_sink=object(),
        )
