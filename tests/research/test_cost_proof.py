from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from lumina_quant.research import cost_proof
from lumina_quant.cli.research import main as cli_main

PROFILE = Path(__file__).parents[2] / "configs/profiles/backtest_cost_realistic.yaml"
MARKET_SHA = "1" * 64
FUNDING_SHA = "2" * 64
RECEIPT_SHA = "3" * 64


def _z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _range(start: datetime, end: datetime) -> dict[str, str]:
    return {"start": _z(start), "end": _z(end)}


def _artifact_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (*cost_proof.EXTERNAL_ARTIFACTS, "verifier_source")
    }


def _router_manifest() -> dict[str, object]:
    starts = [
        datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        datetime(2026, 2, 1, 0, 1, tzinfo=UTC),
        datetime(2026, 3, 1, 0, 1, tzinfo=UTC),
    ]
    folds: list[dict[str, object]] = []
    for index, start in enumerate(starts):
        fold_id = f"fold-{index}"
        selection = {
            "leaves": [
                {
                    "leaf_id": f"leaf-{index}",
                    "traded_symbols": ["BTCUSDT"],
                }
            ]
        }
        variants = []
        for candidate_id in cost_proof.CANDIDATES:
            variants.append(
                {
                    "variant_id": candidate_id,
                    "execution_receipts": [
                        {
                            "leaf_id": f"leaf-{index}",
                            "evaluation_mode": "handler",
                            "engine_source_sha256": "4" * 64,
                            "signal_receipt_sha256": "5" * 64,
                            "position_receipt_sha256": "6" * 64,
                            "engine_receipt_sha256": "7" * 64,
                            "generic_fallback_proxy_count": 0,
                            "current_fold_oos_input_count": 0,
                        }
                    ],
                }
            )
        folds.append(
            {
                "fold_id": fold_id,
                "locked_oos": {
                    "start_utc": _z(start + timedelta(minutes=4)),
                    "end_utc": _z(start + timedelta(minutes=12)),
                },
                "selection": selection,
                "variants": variants,
            }
        )
    return {"folds": folds}


def _market_rows(
    router: dict[str, object],
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], dict[str, object]]]:
    rows: list[dict[str, object]] = []
    indexed: dict[tuple[str, str, str], dict[str, object]] = {}
    for candidate_id, gain in zip(cost_proof.CANDIDATES, (1.0, 0.9), strict=True):
        for router_fold in router["folds"]:
            fold_id = str(router_fold["fold_id"])
            start = datetime.fromisoformat(
                str(router_fold["locked_oos"]["start_utc"]).replace("Z", "+00:00")
            ) - timedelta(minutes=4)
            for grid_index in [0, 1, *range(4, 12)]:
                if grid_index in {0, 4}:
                    prior_mark = mark = 100.0
                elif grid_index == 1:
                    prior_mark, mark = 100.0, 100.0 + gain
                else:
                    prior_mark = 100.0 + (grid_index - 5) * gain
                    mark = prior_mark + gain
                row = {
                    "source_row_id": f"{candidate_id}:{fold_id}:{grid_index}",
                    "artifact_sha256": MARKET_SHA,
                    "symbol": "BTCUSDT",
                    "timestamp": _z(start + timedelta(minutes=grid_index)),
                    "prior_mark_price": prior_mark,
                    "mark_price": mark,
                    "high": max(prior_mark, mark) + 0.5,
                    "low": min(prior_mark, mark) - 0.5,
                }
                rows.append(row)
                indexed[(MARKET_SHA, str(row["source_row_id"]), "BTCUSDT")] = row
    return rows, indexed


def _bindings() -> cost_proof.ExternalBindings:
    router = _router_manifest()
    market_rows, market_index = _market_rows(router)
    source = {
        "schema": "cost_proof_source_data_v1",
        "synthetic_source_count": 0,
        "actual_funding": True,
        "point_in_time_membership": True,
        "post_append_strict_receipt_sha256": RECEIPT_SHA,
        "artifacts": [
            {"kind": "market", "artifact_sha256": MARKET_SHA},
            {"kind": "funding", "artifact_sha256": FUNDING_SHA},
        ],
        "market_rows": market_rows,
        "funding_rows": [],
    }
    return cost_proof.ExternalBindings(
        hashes=_artifact_hashes(),
        profile=yaml.safe_load(PROFILE.read_text(encoding="utf-8")),
        source_manifest=source,
        router_manifest=router,
        membership={},
        trial_ledger={},
        market_artifact_hashes=frozenset({MARKET_SHA}),
        funding_artifact_hashes=frozenset({FUNDING_SHA}),
        market_rows=market_index,
        funding_rows={},
    )


def _scenario(
    candidate_id: str,
    bps: int,
    bindings: cost_proof.ExternalBindings,
    gain: float,
) -> dict[str, object]:
    signals: list[dict[str, object]] = []
    orders: list[dict[str, object]] = []
    fills: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    coefficient = float(bindings.profile["execution"]["slippage_impact_coefficient"])
    router_folds = bindings.router_manifest["folds"]
    for fold_index, router_fold in enumerate(router_folds):
        fold_id = str(router_fold["fold_id"])
        start = datetime.fromisoformat(
            str(router_fold["locked_oos"]["start_utc"]).replace("Z", "+00:00")
        ) - timedelta(minutes=4)
        period_indexes = [0, 1, *range(4, 12)]
        validation_stop = f"{candidate_id}:stop:{fold_id}:validation"
        locked_stop = f"{candidate_id}:stop:{fold_id}:locked"
        stops = [
            {
                "stop_id": validation_stop,
                "symbol": "BTCUSDT",
                "side": "SELL",
                "quantity": 1.0,
                "stop_price": 90.0,
                "source": "engine_default",
                "activated_period_id": f"{fold_id}:p0",
                "deactivated_period_id": f"{fold_id}:p1",
                "trigger_fill_id": None,
            },
            {
                "stop_id": locked_stop,
                "symbol": "BTCUSDT",
                "side": "SELL",
                "quantity": 1.0,
                "stop_price": 90.0,
                "source": "engine_default",
                "activated_period_id": f"{fold_id}:p4",
                "deactivated_period_id": f"{fold_id}:p11",
                "trigger_fill_id": None,
            },
        ]
        prior_equity = 1_000.0
        periods: list[dict[str, object]] = []
        for grid_index in period_indexes:
            period_id = f"{fold_id}:p{grid_index}"
            timestamp = _z(start + timedelta(minutes=grid_index))
            segment = "validation" if grid_index < 2 else "locked_oos"
            entry = grid_index in {0, 4}
            start_position = 0.0 if entry else 1.0
            position = 1.0
            if grid_index in {0, 4}:
                prior_mark = mark = 100.0
            elif grid_index == 1:
                prior_mark, mark = 100.0, 100.0 + gain
            else:
                prior_mark = 100.0 + (grid_index - 5) * gain
                mark = prior_mark + gain
            low = min(prior_mark, mark) - 0.5
            high = max(prior_mark, mark) + 0.5
            gross = start_position * (mark - prior_mark)
            signals.append(
                {
                    "period_id": period_id,
                    "timestamp": timestamp,
                    "symbol": "BTCUSDT",
                    "signal": 1.0,
                    "start_position": start_position,
                    "position": position,
                    "prior_mark_price": prior_mark,
                    "mark_price": mark,
                    "high": high,
                    "low": low,
                    "gross_pnl": gross,
                    "market_data_artifact_sha256": MARKET_SHA,
                    "market_source_row_id": f"{candidate_id}:{fold_id}:{grid_index}",
                }
            )
            linear = 0.0
            impact = 0.0
            active_stop = validation_stop if segment == "validation" else locked_stop
            if entry:
                order_id = f"{candidate_id}:order:{fold_id}:{grid_index}"
                fill_id = f"{candidate_id}:fill:{fold_id}:{grid_index}"
                requested = 1.0
                price = 100.0
                signed_quote = requested * price
                bar_volume = 1_000_000_000.0
                adv = bar_volume * price
                participation = signed_quote / adv
                impact_rate = coefficient * math.sqrt(participation)
                impact = signed_quote * impact_rate
                linear = signed_quote * bps / 10_000
                orders.append(
                    {
                        "order_id": order_id,
                        "period_id": period_id,
                        "timestamp": timestamp,
                        "symbol": "BTCUSDT",
                        "signed_qty": requested,
                        "signed_quote_notional": signed_quote,
                        "requested_qty": requested,
                        "direction": "BUY",
                        "order_type": "IOC",
                        "time_in_force": "IOC",
                        "is_maker": False,
                        "is_entry": True,
                        "protective_stop_id": active_stop,
                    }
                )
                fills.append(
                    {
                        "fill_id": fill_id,
                        "order_id": order_id,
                        "period_id": period_id,
                        "timestamp": timestamp,
                        "symbol": "BTCUSDT",
                        "is_entry": True,
                        "signed_qty": requested,
                        "requested_qty": requested,
                        "direction": "BUY",
                        "fill_price": price,
                        "signed_quote_notional": signed_quote,
                        "is_maker": False,
                        "bar_volume": bar_volume,
                        "observed_adv_quote": adv,
                        "participation": participation,
                        "impact_coefficient": coefficient,
                        "sqrt_impact_rate": impact_rate,
                        "sqrt_impact_cash_cost": impact,
                        "protective_stop_id": active_stop,
                        "protective_stop_source": "engine_default",
                    }
                )
            funding = 0.0
            net = gross - linear - impact + funding
            equity = prior_equity + net
            exposure = max(abs(start_position * prior_mark), abs(position * mark)) / prior_equity
            raw_return = net / prior_equity
            normalized = raw_return / exposure
            worst_equity = prior_equity + start_position * (low - prior_mark)
            maintenance = abs(start_position * low) * (0.005 + 0.0005)
            periods.append(
                {
                    "period_id": period_id,
                    "timestamp": timestamp,
                    "segment": segment,
                    "expected_funding": [],
                    "gross_pnl": gross,
                    "linear_cost": linear,
                    "impact_cost": impact,
                    "funding_cashflow": funding,
                    "net_pnl": net,
                    "prior_equity": prior_equity,
                    "equity": equity,
                    "gross_exposure_fraction": exposure,
                    "raw_net_return": raw_return,
                    "exposure_normalized_net_return": normalized,
                    "position_notional": position * mark,
                    "active_protective_stop_ids": [active_stop],
                    "worst_intrabar_equity": worst_equity,
                    "maintenance_margin_required": maintenance,
                }
            )
            prior_equity = equity
        receipts = next(
            variant["execution_receipts"]
            for variant in router_fold["variants"]
            if variant["variant_id"] == candidate_id
        )
        folds.append(
            {
                "fold_id": fold_id,
                "router_execution_receipts_sha256": cost_proof._canonical_sha256(receipts),
                "bar_interval_seconds": 60,
                "evaluated_range": _range(start, start + timedelta(minutes=12)),
                "validation_range": _range(start, start + timedelta(minutes=2)),
                "locked_oos_range": _range(
                    start + timedelta(minutes=4), start + timedelta(minutes=12)
                ),
                "purge": {
                    "expected_count": 1,
                    "removed_range": _range(
                        start + timedelta(minutes=2), start + timedelta(minutes=3)
                    ),
                    "removed_rows": [
                        {
                            "period_id": f"{fold_id}:purge",
                            "timestamp": _z(start + timedelta(minutes=2)),
                        }
                    ],
                },
                "embargo": {
                    "expected_count": 1,
                    "removed_range": _range(
                        start + timedelta(minutes=3), start + timedelta(minutes=4)
                    ),
                    "removed_rows": [
                        {
                            "period_id": f"{fold_id}:embargo",
                            "timestamp": _z(start + timedelta(minutes=3)),
                        }
                    ],
                },
                "initial_equity": 1_000.0,
                "periods": periods,
                "funding": [],
                "protective_stops": stops,
                "entry_count": 2,
                "protective_stop_count": 2,
                "liquidation_count": 0,
                "ruin": False,
                "equity": prior_equity,
            }
        )
    scenario = {
        "cost_bps": bps,
        "evaluation_modes": ["handler"],
        "generic_fallback_proxy_count": 0,
        "current_fold_oos_input_count": 0,
        "router_replay_manifest_sha256": bindings.hashes["router_replay_manifest"],
        "membership_sha256": bindings.hashes["membership"],
        "signal_position_tape": signals,
        "orders": orders,
        "fills": fills,
        "signal_tape_sha256": cost_proof._canonical_sha256(signals),
        "order_tape_sha256": cost_proof._canonical_sha256(orders),
        "execution_tape_sha256": cost_proof._canonical_sha256(fills),
        "economic_tape_sha256": cost_proof._canonical_sha256(cost_proof._economic_tape(folds)),
        "folds": folds,
    }
    return scenario


def _bundle() -> tuple[dict[str, object], cost_proof.ExternalBindings]:
    bindings = _bindings()
    candidates = []
    for candidate_id, gain in zip(cost_proof.CANDIDATES, (1.0, 0.9), strict=True):
        candidates.append(
            {
                "candidate_id": candidate_id,
                "router_replay_manifest_sha256": bindings.hashes["router_replay_manifest"],
                "membership_sha256": bindings.hashes["membership"],
                "scenarios": [
                    _scenario(candidate_id, bps, bindings, gain) for bps in cost_proof.COST_LADDER
                ],
            }
        )
    twenty = [candidate["scenarios"][2] for candidate in candidates]
    locked_ids = [
        period["period_id"]
        for period in twenty[0]["folds"][0]["periods"]
        if period["segment"] == "locked_oos"
    ]
    locked_ids = [
        period["period_id"]
        for fold in twenty[0]["folds"]
        for period in fold["periods"]
        if period["segment"] == "locked_oos"
    ]
    normalized = [
        [
            period["exposure_normalized_net_return"]
            for fold in scenario["folds"]
            for period in fold["periods"]
            if period["segment"] == "locked_oos"
        ]
        for scenario in twenty
    ]
    selection = {
        "candidate_ids": list(cost_proof.CANDIDATES),
        "post_oos_research_variant": True,
        "current_fold_oos_input_count": 0,
    }
    dedup = {
        "input_trial_count": 2,
        "effective_trial_count": 2,
        "current_fold_oos_input_count": 0,
    }
    trial_ledger = {
        "schema": "cost_proof_trial_ledger_v1",
        "cost_bps": 20,
        "trial_ids": list(cost_proof.CANDIDATES),
        "locked_oos_period_ids": locked_ids,
        "normalized_returns_20bp": normalized,
        "raw_trial_count": 2,
        "effective_trial_count": 2,
        "current_fold_oos_input_count": 0,
        "selection_receipt": selection,
        "selection_receipt_sha256": cost_proof._canonical_sha256(selection),
        "dedup_receipt": dedup,
        "dedup_receipt_sha256": cost_proof._canonical_sha256(dedup),
    }
    bindings = replace(bindings, trial_ledger=trial_ledger)
    provenance = {
        field: bindings.hashes[name] for name, field in cost_proof.PROVENANCE_ARTIFACTS.items()
    }
    provenance["candidate_ids_sha256"] = cost_proof.candidate_ids_sha256()
    return (
        {
            "schema": cost_proof.SCHEMA,
            "candidate_ids": list(cost_proof.CANDIDATES),
            "cost_ladder_bps": list(cost_proof.COST_LADDER),
            "cscv_splits": cost_proof.CSCV_SPLITS,
            "provenance": provenance,
            "candidates": candidates,
        },
        bindings,
    )


def _rehash_scenario(scenario: dict[str, object]) -> None:
    scenario["signal_tape_sha256"] = cost_proof._canonical_sha256(scenario["signal_position_tape"])
    scenario["order_tape_sha256"] = cost_proof._canonical_sha256(scenario["orders"])
    scenario["execution_tape_sha256"] = cost_proof._canonical_sha256(scenario["fills"])
    scenario["economic_tape_sha256"] = cost_proof._canonical_sha256(
        cost_proof._economic_tape(scenario["folds"])
    )


def _passing_statistics(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    trial_counts: list[float] = []
    monkeypatch.setattr(cost_proof, "cscv_pbo", lambda *args, **kwargs: 0.1)
    monkeypatch.setattr(cost_proof, "spa_like_pvalue", lambda *args, **kwargs: 0.01)

    def dsr(*args: object, **kwargs: object) -> float:
        trial_counts.append(float(kwargs["num_trials"]))
        return 0.95

    monkeypatch.setattr(cost_proof, "deflated_sharpe_ratio", dsr)
    return trial_counts


def test_authenticated_engine_ledger_passes_with_raw_trial_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, bindings = _bundle()
    trial_counts = _passing_statistics(monkeypatch)

    report = cost_proof.evaluate_cost_proof(evidence, bindings=bindings)

    assert report.status == "PASS"
    assert report.selected_candidate_id == cost_proof.CANDIDATES[0]
    assert trial_counts == [2.0, 2.0]
    assert all(item["metrics"]["effective_trial_count"] == 2.0 for item in report.candidate_reports)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda scenario: scenario["folds"][0]["periods"][0].__setitem__(
            "gross_exposure_fraction", 0.5
        ),
        lambda scenario: scenario["fills"][0].__setitem__("sqrt_impact_cash_cost", 0.0),
        lambda scenario: scenario["signal_position_tape"][0].__setitem__("position", 2.0),
        lambda scenario: scenario["folds"][0]["periods"][0].__setitem__(
            "active_protective_stop_ids", []
        ),
        lambda scenario: scenario["folds"][0]["periods"][1].__setitem__(
            "worst_intrabar_equity", 1.0
        ),
        lambda scenario: scenario.__setitem__("generic_fallback_proxy_count", False),
    ],
)
def test_derived_engine_fields_fail_closed(monkeypatch: pytest.MonkeyPatch, mutate: object) -> None:
    evidence, bindings = _bundle()
    scenario = evidence["candidates"][0]["scenarios"][0]
    mutate(scenario)
    _rehash_scenario(scenario)
    _passing_statistics(monkeypatch)

    report = cost_proof.evaluate_cost_proof(evidence, bindings=bindings)

    assert report.status == "STOP"


def test_trial_ledger_cannot_undercount_search_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, bindings = _bundle()
    ledger = deepcopy(bindings.trial_ledger)
    ledger["raw_trial_count"] = 1
    bindings = replace(bindings, trial_ledger=ledger)
    _passing_statistics(monkeypatch)

    report = cost_proof.evaluate_cost_proof(evidence, bindings=bindings)

    assert report.status == "STOP"
    assert report.reasons == ("invalid whole-search trial ledger",)


def test_scientific_gate_failure_is_reject_not_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, bindings = _bundle()
    monkeypatch.setattr(cost_proof, "cscv_pbo", lambda *args, **kwargs: 0.1)
    monkeypatch.setattr(cost_proof, "spa_like_pvalue", lambda *args, **kwargs: 0.01)
    monkeypatch.setattr(cost_proof, "deflated_sharpe_ratio", lambda *args, **kwargs: 0.2)

    report = cost_proof.evaluate_cost_proof(evidence, bindings=bindings)

    assert report.status == "REJECT"
    assert all(item["status"] == "REJECT" for item in report.candidate_reports)


def test_mdd_includes_initial_capital_peak() -> None:
    assert cost_proof.max_drawdown(
        np.asarray([-0.1, 0.1], dtype=float)
    ) == pytest.approx(0.1)


def test_duplicate_json_and_unknown_fields_stop() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        cost_proof._json_bytes(b'{"schema":"x","schema":"y"}')
    evidence, bindings = _bundle()
    evidence["unexpected"] = True
    assert cost_proof.evaluate_cost_proof(evidence, bindings=bindings).status == "STOP"


def test_missing_external_bindings_stop() -> None:
    evidence, _ = _bundle()
    assert cost_proof.evaluate_cost_proof(evidence).status == "STOP"
    assert cost_proof.evaluate_cost_proof_file("missing.json", PROFILE).status == "STOP"


def test_artifact_binding_hashes_external_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _bindings().source_manifest
    files: dict[str, Path] = {}
    for name in cost_proof.EXTERNAL_ARTIFACTS:
        path = tmp_path / name
        if name == "profile":
            path.write_bytes(PROFILE.read_bytes())
        elif name == "source_data_manifest":
            path.write_text(json.dumps(source), encoding="utf-8")
        elif name in {"lifecycle", "membership", "router_replay_manifest", "trial_ledger"}:
            path.write_text("{}", encoding="utf-8")
        else:
            path.write_text(name, encoding="utf-8")
        files[name] = path
    monkeypatch.setattr(cost_proof, "validate_symbol_lifecycle_registry", lambda value: {})
    monkeypatch.setattr(cost_proof, "validate_fold_membership_manifest", lambda registry, value: {})
    monkeypatch.setattr(
        cost_proof,
        "evaluate_router_replay",
        lambda *args, **kwargs: SimpleNamespace(status="PASS"),
    )

    bindings = cost_proof._artifact_bindings(files)

    assert (
        bindings.hashes["producer_source"]
        == hashlib.sha256(files["producer_source"].read_bytes()).hexdigest()
    )
    assert bindings.market_artifact_hashes == frozenset({MARKET_SHA})


def test_duplicate_profile_key_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text("profile: backtest_cost_realistic\nprofile: duplicate\n", encoding="utf-8")
    files = {name: tmp_path / name for name in cost_proof.EXTERNAL_ARTIFACTS}
    for name, path in files.items():
        if name == "profile":
            files[name] = profile
        else:
            path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate YAML key"):
        cost_proof._artifact_bindings(files)


def test_cli_passes_every_external_binding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def evaluate(input_path: str, profile_path: str, **kwargs: object) -> object:
        captured.update({"input_path": input_path, "profile_path": profile_path, **kwargs})
        return SimpleNamespace(status="PASS", to_json=lambda: '{"status":"PASS"}')

    monkeypatch.setattr(cost_proof, "evaluate_cost_proof_file", evaluate)
    arguments = [
        "cost-proof",
        "--input",
        "proof.json",
        "--config",
        "profile.yaml",
        "--source-data-manifest",
        "source.json",
        "--router-replay-manifest",
        "router.json",
        "--router-source-artifact",
        "router-source.json",
        "--lifecycle",
        "lifecycle.json",
        "--membership",
        "membership.json",
        "--trial-ledger",
        "trials.json",
        "--producer-source",
        "producer.py",
        "--commit-receipt",
        "commit.txt",
        "--router-producer-source",
        "router-producer.py",
        "--router-commit-receipt",
        "router-commit.txt",
    ]

    assert cli_main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "PASS"}
    assert captured == {
        "input_path": "proof.json",
        "profile_path": "profile.yaml",
        "source_data_manifest_path": "source.json",
        "router_replay_manifest_path": "router.json",
        "router_source_artifact_path": "router-source.json",
        "lifecycle_path": "lifecycle.json",
        "membership_path": "membership.json",
        "trial_ledger_path": "trials.json",
        "producer_source_path": "producer.py",
        "commit_receipt_path": "commit.txt",
        "router_producer_source_path": "router-producer.py",
        "router_commit_receipt_path": "router-commit.txt",
    }


def test_market_marks_must_match_external_source_rows() -> None:
    evidence, bindings = _bundle()
    scenario = evidence["candidates"][0]["scenarios"][0]
    scenario["signal_position_tape"][0]["market_source_row_id"] = "missing"
    _rehash_scenario(scenario)

    assert cost_proof.evaluate_cost_proof(evidence, bindings=bindings).status == "STOP"


def test_funding_obligations_are_derived_and_source_bound() -> None:
    _, bindings = _bundle()
    boundary = "2026-01-01T00:00:00Z"
    period = {
        "period_id": "p0",
        "timestamp": boundary,
        "expected_funding": [],
    }
    fold = {"periods": [period], "funding": []}
    times = {"p0": datetime(2026, 1, 1, tzinfo=UTC)}
    tape = {"signals": {("p0", "BTCUSDT"): {"prior_mark_price": 100.0}}}
    positions = {"p0": ({"BTCUSDT": 1.0}, {"BTCUSDT": 1.0})}

    assert cost_proof._strict_funding(fold, times, tape, positions, bindings) is None

    period["expected_funding"] = [{"symbol": "BTCUSDT", "boundary": boundary}]
    source = {
        "source_row_id": "funding-0",
        "artifact_sha256": FUNDING_SHA,
        "symbol": "BTCUSDT",
        "boundary": boundary,
        "observed_rate": 0.001,
    }
    fold["funding"] = [
        {
            "period_id": "p0",
            "symbol": "BTCUSDT",
            "settlement_id": "settlement-0",
            "source_row_id": "funding-0",
            "source_artifact_sha256": FUNDING_SHA,
            "boundary": boundary,
            "observed_rate": 0.001,
            "signed_open_notional": 100.0,
            "signed_cashflow": -0.1,
        }
    ]
    bindings = replace(
        bindings,
        funding_rows={(FUNDING_SHA, "funding-0", "BTCUSDT"): source},
    )

    assert cost_proof._strict_funding(fold, times, tape, positions, bindings) == {
        "p0": pytest.approx(-0.1)
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bar_interval_seconds", 120),
        ("initial_equity", 2_000.0),
    ],
)
def test_grid_and_initial_equity_are_recomputed(field: str, value: float) -> None:
    evidence, bindings = _bundle()
    scenario = evidence["candidates"][0]["scenarios"][0]
    scenario["folds"][0][field] = value
    _rehash_scenario(scenario)

    assert cost_proof.evaluate_cost_proof(evidence, bindings=bindings).status == "STOP"


def test_trial_ledger_period_identity_is_exact() -> None:
    evidence, bindings = _bundle()
    ledger = deepcopy(bindings.trial_ledger)
    ledger["locked_oos_period_ids"][0] = "other-period"
    bindings = replace(bindings, trial_ledger=ledger)

    assert cost_proof.evaluate_cost_proof(evidence, bindings=bindings).status == "STOP"
