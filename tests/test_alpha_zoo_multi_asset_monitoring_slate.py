from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_multi_asset_monitoring_slate.py"
SPEC = importlib.util.spec_from_file_location(
    "run_alpha_zoo_multi_asset_monitoring_slate",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _candidate(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model_id": "candidate",
        "symbol": "SOLUSDT",
        "asset_group": "crypto_high_beta_alt",
        "timeframe": "2h",
        "family": "relative_strength_chandelier_breakout",
        "side": "long_short",
        "leverage": 4.0,
        "allocation_fraction": 0.125,
        "notional_fraction": 0.5,
        "train_return": 0.30,
        "validation_return": 0.12,
        "locked_oos_return": 0.04,
        "train_mdd": 0.08,
        "validation_mdd": 0.05,
        "locked_oos_mdd": 0.03,
        "train_trade_event_count": 180,
        "validation_trade_event_count": 40,
        "locked_oos_trade_event_count": 24,
        "train_return_per_turnover_proxy_bps": 28.0,
        "validation_return_per_turnover_proxy_bps": 42.0,
        "locked_oos_return_per_turnover_proxy_bps": 22.0,
        "paper_candidate_gate_pass": True,
        "primary_10bps_promotion_gate_pass": True,
        "execution_efficiency_proxy_gate_pass": True,
        "ready_for_paper": True,
        "ready_for_real": False,
        "real_money_execution": False,
        "replay_live_notional_parity": "recorded",
        "locked_oos_liquidation_count": 0,
        "locked_oos_account_wipeout_count": 0,
        "rejection_reasons": [],
    }
    row.update(overrides)
    return row


def _source_payload(
    *,
    artifact_kind: str,
    symbols: list[str],
    paper: list[dict[str, object]] | None = None,
    top: list[dict[str, object]] | None = None,
    shadows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "artifact_kind": artifact_kind,
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_testnet_only": True,
        "source_data": {
            "symbols": symbols,
            "timeframes": ["1h", "2h"],
        },
        "asset_groups": {
            "crypto_high_beta_alt": ["SOLUSDT", "DOGEUSDT"],
            "crypto_major": ["ETHUSDT", "BTCUSDT"],
            "crypto_payment_alt": ["XRPUSDT"],
            "crypto_exchange_beta": ["BNBUSDT"],
        },
        "paper_testnet_handoff": {
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
            "candidates": paper or [],
        },
        "top_candidates": top or [],
        "no_promotion_shadow_shortlist": {
            "ready_for_real": False,
            "real_money_execution": False,
            "shadows": shadows or [],
        },
    }


def _loaded(label: str, payload: dict[str, object], tmp_path: Path) -> object:
    return MODULE.LoadedSource(label=label, path=tmp_path / f"{label}.json", payload=payload)


def test_monitoring_score_ignores_locked_oos() -> None:
    strong_oos = _candidate(locked_oos_return=0.80, locked_oos_return_per_turnover_proxy_bps=120.0)
    weak_oos = _candidate(locked_oos_return=-0.80, locked_oos_return_per_turnover_proxy_bps=-120.0)

    assert MODULE._monitoring_score(strong_oos) == MODULE._monitoring_score(weak_oos)


def test_no_real_money_guard_fails_closed_on_nested_candidate(tmp_path: Path) -> None:
    payload = _source_payload(
        artifact_kind="bad_source",
        symbols=["SOLUSDT"],
        paper=[_candidate(ready_for_real=True)],
    )
    source = _loaded("bad", payload, tmp_path)

    with pytest.raises(ValueError, match="real-money guard violation"):
        MODULE._assert_no_real_money_disabled(source)


def test_build_payload_keeps_all_symbols_not_only_top_one_or_two(tmp_path: Path) -> None:
    sol = _candidate(model_id="sol-paper", symbol="SOLUSDT")
    eth = _candidate(
        model_id="eth-paper",
        symbol="ETHUSDT",
        asset_group="crypto_major",
        family="relative_residual_reclaim",
        validation_return=0.05,
    )
    xrp_shadow = _candidate(
        model_id="xrp-shadow",
        symbol="XRPUSDT",
        asset_group="crypto_payment_alt",
        paper_candidate_gate_pass=False,
        primary_10bps_promotion_gate_pass=False,
        execution_efficiency_proxy_gate_pass=False,
        ready_for_paper=False,
        locked_oos_trade_event_count=0,
        rejection_reasons=["locked_oos_trade_event_count_0_below_20"],
    )
    bnb_shadow = _candidate(
        model_id="bnb-shadow",
        symbol="BNBUSDT",
        asset_group="crypto_exchange_beta",
        paper_candidate_gate_pass=False,
        primary_10bps_promotion_gate_pass=False,
        execution_efficiency_proxy_gate_pass=False,
        ready_for_paper=False,
        validation_return=0.03,
        rejection_reasons=["validation_mdd_0.1300_above_0.12"],
    )
    sources = [
        _loaded(
            "source_a",
            _source_payload(
                artifact_kind="source_a_artifact",
                symbols=["SOLUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT"],
                paper=[sol, eth],
                shadows=[xrp_shadow],
            ),
            tmp_path,
        ),
        _loaded(
            "source_b",
            _source_payload(
                artifact_kind="source_b_artifact",
                symbols=["BNBUSDT"],
                top=[bnb_shadow],
            ),
            tmp_path,
        ),
    ]

    payload = MODULE.build_payload_from_loaded_sources(
        sources,
        output_dir=tmp_path,
        write_outputs=True,
    )

    matrix = {row["symbol"]: row for row in payload["asset_monitoring_matrix"]}
    assert {"SOLUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"} <= set(matrix)
    assert matrix["SOLUSDT"]["paper_monitor_count"] == 1
    assert matrix["ETHUSDT"]["paper_monitor_count"] == 1
    assert matrix["XRPUSDT"]["coverage_blocked_shadow_count"] == 1
    assert matrix["BNBUSDT"]["shadow_watchlist_count"] == 1
    assert matrix["DOGEUSDT"]["total_candidate_rows"] == 0
    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False
    assert Path(payload["output_paths"]["latest_json"]).exists()
    assert Path(payload["output_paths"]["asset_monitoring_matrix_csv"]).exists()


def test_paper_handoff_groups_multiple_symbols_and_remains_real_disabled(tmp_path: Path) -> None:
    payload = MODULE.build_payload_from_loaded_sources(
        [
            _loaded(
                "source",
                _source_payload(
                    artifact_kind="source_artifact",
                    symbols=["SOLUSDT", "ETHUSDT"],
                    paper=[
                        _candidate(model_id="sol-paper", symbol="SOLUSDT"),
                        _candidate(
                            model_id="eth-paper",
                            symbol="ETHUSDT",
                            asset_group="crypto_major",
                        ),
                    ],
                ),
                tmp_path,
            )
        ],
        output_dir=tmp_path,
        write_outputs=False,
    )

    handoff = payload["paper_monitoring_handoff"]
    assert set(handoff["candidates_by_symbol"]) == {"SOLUSDT", "ETHUSDT"}
    assert handoff["ready_for_real"] is False
    assert handoff["real_money_execution"] is False
    assert handoff["real_execution_allowed"] is False
    assert handoff["preflight"]["monitor_all_symbols_together"] is True


def test_coverage_blocked_shadow_reason_mapping() -> None:
    status, action, reasons = MODULE._monitoring_status_and_action(
        {
            "paper_candidate_gate_pass": False,
            "rejection_reasons": ["locked_oos_trade_event_count_0_below_20"],
        }
    )

    assert status == "coverage_blocked_shadow"
    assert action == "extend_locked_oos_data_coverage_before_any_paper_review"
    assert reasons == ["locked_oos_trade_event_count_0_below_20"]
