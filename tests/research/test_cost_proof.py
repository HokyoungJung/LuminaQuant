from __future__ import annotations

from pathlib import Path

import pytest

from lumina_quant.cli.research import main as cli_main
from lumina_quant.research import cost_proof


SHA = "a" * 64
ROOTS = {
    "source_data_commit_sha256": "b" * 64,
    "search_run_receipt_sha256": "c" * 64,
    "cost_proof_commit_sha256": "d" * 64,
    "router_source_artifact_sha256": "e" * 64,
    "router_commit_receipt_sha256": "f" * 64,
}


def _argv(*extra: str) -> list[str]:
    return [
        "cost-proof",
        "--input",
        "evidence.json",
        "--config",
        "profile.yaml",
        "--source-data-manifest",
        "source.json",
        "--source-run-receipt",
        "source-receipt.json",
        "--search-run-receipt",
        "search-receipt.json",
        "--router-replay-manifest",
        "router.json",
        "--router-source-artifact",
        "router.py",
        "--lifecycle",
        "lifecycle.json",
        "--membership",
        "membership.json",
        "--trial-ledger",
        "trials.json",
        "--producer-source",
        "producer.py",
        "--commit-receipt",
        "cost-receipt.json",
        "--router-producer-source",
        "router-producer.py",
        "--router-commit-receipt",
        "router-receipt.json",
        "--source-data-commit-sha256",
        ROOTS["source_data_commit_sha256"],
        "--search-run-receipt-sha256",
        ROOTS["search_run_receipt_sha256"],
        "--cost-proof-commit-sha256",
        ROOTS["cost_proof_commit_sha256"],
        "--router-source-artifact-sha256",
        ROOTS["router_source_artifact_sha256"],
        "--router-commit-receipt-sha256",
        ROOTS["router_commit_receipt_sha256"],
        *extra,
    ]


def test_cli_forwards_ordered_v2_artifacts_and_all_trusted_roots(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def evaluate(*args: object, **kwargs: object) -> cost_proof.CostProofReport:
        captured["args"] = args
        captured.update(kwargs)
        return cost_proof.CostProofReport("PASS", "cost_proof_v2", (), (), None)

    monkeypatch.setattr(cost_proof, "evaluate_cost_proof_file", evaluate)
    assert (
        cli_main(
            _argv(
                "--market-artifact",
                f"{SHA}=market=one.json",
                "--market-artifact",
                f"{'1' * 64}=market-two.json",
                "--funding-artifact",
                f"{'2' * 64}=funding.json",
                "--router-artifact",
                f"{'3' * 64}=router-tape.json",
                "--trial-result-artifact",
                f"{'4' * 64}=trial.json",
            )
        )
        == 0
    )
    assert captured["args"] == ("evidence.json", "profile.yaml")
    assert captured["market_artifact_paths"] == {
        SHA: "market=one.json",
        "1" * 64: "market-two.json",
    }
    assert captured["funding_artifact_paths"] == {"2" * 64: "funding.json"}
    assert captured["router_artifact_paths"] == {"3" * 64: "router-tape.json"}
    assert captured["trial_result_artifact_paths"] == {"4" * 64: "trial.json"}
    assert captured["trusted_roots"] == ROOTS
    assert '"status":"PASS"' in capsys.readouterr().out


@pytest.mark.parametrize("status, expected_exit", [("PASS", 0), ("REJECT", 1), ("STOP", 2)])
def test_cli_preserves_cost_proof_exit_codes(
    monkeypatch: pytest.MonkeyPatch, status: str, expected_exit: int
) -> None:
    monkeypatch.setattr(
        cost_proof,
        "evaluate_cost_proof_file",
        lambda *args, **kwargs: cost_proof.CostProofReport(status, "cost_proof_v2", (), (), None),
    )
    assert cli_main(_argv()) == expected_exit


@pytest.mark.parametrize(
    "binding",
    [
        "not-a-binding",
        f"{'A' * 64}=uppercase.json",
        "a" * 63 + "=short.json",
        "=" + str(Path("empty-digest.json")),
    ],
)
def test_cli_malformed_artifact_binding_fails_closed(
    binding: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli_main(_argv("--market-artifact", binding)) == 2
    assert capsys.readouterr().out == (
        '{"candidate_reports":[],"reasons":["invalid artifact binding or trusted root"],'
        '"selected_candidate_id":null,"status":"STOP","version":"cost_proof_v2"}\n'
    )


def test_cli_duplicate_artifact_digest_fails_before_evaluator(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cost_proof,
        "evaluate_cost_proof_file",
        lambda *args, **kwargs: pytest.fail("duplicate binding reached evaluator"),
    )
    assert (
        cli_main(
            _argv("--market-artifact", f"{SHA}=one.json", "--market-artifact", f"{SHA}=two.json")
        )
        == 2
    )
    assert '"status":"STOP"' in capsys.readouterr().out


from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
import numpy as np

from tests.research.test_router_replay import _bundle


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()


def _cost_profile() -> dict[str, Any]:
    return {
        "profile": "backtest_cost_realistic",
        "research": {
            "strict_selection_gate": True,
            "use_lockbox_split": True,
            "purge_embargo_bars": 1,
            "single_correlation_discount": True,
            "hac_inference": True,
            "cscv_pbo": True,
            "exposure_normalized_promotion": True,
            "enforce_selection_reject_gate": True,
            "dsr_gate_floor": 0.90,
            "spa_gate_ceiling": 0.05,
            "pbo_gate_ceiling": 0.50,
            "max_cross_trial_pbo": 0.50,
            "route_unmapped_registered_strategies": True,
            "require_actual_engine_routing": True,
            "emit_candidate_overfit_stats": True,
            "portfolio_honest_gate": True,
        },
        "execution": {
            "slippage_impact_model": "sqrt_impact",
            "slippage_impact_coefficient": 0.10,
            "slippage_adv_quote": 0.0,
            "require_funding_coverage": True,
            "funding_on_utc_boundary": True,
            "funding_interval_hours": 8,
            "maintenance_margin_rate": 0.005,
            "liquidation_buffer_rate": 0.0005,
        },
        "risk": {
            "default_stop_loss_pct": 0.01,
            "attach_default_protective_stop": True,
            "enforce_order_risk_gate_in_backtest": True,
        },
        "data": {"kinds": ["market", "funding"]},
        "backtest": {"leverage": 3},
        "live": {
            "mode": "paper",
            "testnet": True,
            "require_real_enable_flag": True,
            "allow_market_orders": False,
            "shadow_live_enabled": False,
        },
    }


def _row_hashes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sequence_id": row["sequence_id"],
            "fold_id": row["fold_id"],
            "variant_id": row["variant_id"],
            "leaf_id": row["leaf_id"],
            "engine_receipt_sha256": row["engine_receipt_sha256"],
            "row_sha256": _canonical_sha256(row),
        }
        for row in rows
    ]


def _cost_fold(
    candidate_id: str,
    fold_number: int,
    cost_bps: int,
    start: datetime,
    market_rows: dict[tuple[str, str, str], dict[str, Any]],
    funding_rows: dict[tuple[str, str, str], dict[str, Any]],
    receipt_hash: str,
    *,
    bar_interval_hours: int = 6,
    leaf_id: str | None = None,
    market_artifact: str | None = None,
    funding_artifact: str | None = None,
    fold_id: str | None = None,
    prices_override: list[float] | None = None,
    entry_indices: tuple[int, ...] = (0, 4),
    flatten_indices: tuple[int, ...] = (3, 11),
    trade_quantity: float = 1.0,
    trade_quantities: dict[int, float] | None = None,
    bar_volume_base: float = 100.0,
    symbol: str = "BTC-USD",
    router_execution_receipts_sha256: str | None = None,
    terminal_liquidation: bool = False,
    stop_source: str = "engine_default",
    stop_price: float | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build one exact validation/purge/embargo/lockbox cell and its engine tapes."""
    leaf = leaf_id or f"leaf-{fold_number}"
    fold_name = fold_id or f"fold-{fold_number}"
    primary = candidate_id == cost_proof.CANDIDATES[0]
    artifact = market_artifact or ("1" if primary else "6") * 64
    funding_artifact = funding_artifact or "2" * 64
    times = [start + timedelta(hours=bar_interval_hours * index) for index in range(15)]

    def stamp(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")

    period_times = [times[index] for index in (*range(4), *range(6, 14))]
    period_ids = [f"{fold_name}-p{index}" for index in range(12)]
    segments = ["validation"] * 4 + ["locked_oos"] * 8
    prices = prices_override or (
        [100.0, 110.0, 120.0, 130.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]
        if primary
        else [100.0, 101.0, 100.0, 102.0, 100.0, 101.0, 100.0, 102.0, 101.0, 103.0, 102.0, 104.0]
    )
    signals: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    periods: list[dict[str, Any]] = []
    funding: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    position, cash, basis = 0.0, 100_000.0, 0.0
    active_stop: str | None = None
    for index, (period_id, when, segment, mark) in enumerate(
        zip(period_ids, period_times, segments, prices, strict=True)
    ):
        prior = prices[index - 1] if index else mark
        low, high = min(prior, mark), max(prior, mark)
        row_id = f"m-{fold_number}-{index}"
        market_rows[(artifact, row_id, symbol)] = {
            "timestamp": stamp(when),
            "prior_mark_price": prior,
            "mark_price": mark,
            "high": high,
            "low": low,
            "price_tick_size": 1.0,
            "quantity_step_size": 1.0,
            "bar_volume_base": bar_volume_base,
        }
        sequence = f"{candidate_id[:8]}-{fold_number}-{index}"
        signal = {
            "cost_bps": cost_bps,
            "sequence_id": sequence,
            "fold_id": fold_name,
            "variant_id": candidate_id,
            "leaf_id": leaf,
            "engine_receipt_sha256": receipt_hash,
            "period_id": period_id,
            "timestamp": stamp(when),
            "symbol": symbol,
            "signal": 1.0,
            "start_position": position,
            "position": 0.0,
            "prior_mark_price": prior,
            "mark_price": mark,
            "high": high,
            "low": low,
            "gross_pnl": position * (mark - prior),
            "market_data_artifact_sha256": artifact,
            "market_source_row_id": row_id,
        }
        signals.append(signal)
        expected_funding: list[dict[str, str]] = []
        funding_cash = 0.0
        if when.hour in {0, 8, 16} and position:
            expected_funding = [{"symbol": symbol, "boundary": stamp(when)}]
            funding_id = f"u-{fold_number}-{index}"
            rate = 0.0001
            funding_rows[(funding_artifact, funding_id, symbol)] = {
                "boundary": stamp(when),
                "observed_rate": rate,
            }
            funding_cash = -(position * prior) * rate
            funding.append(
                {
                    "period_id": period_id,
                    "symbol": symbol,
                    "settlement_id": f"s-{fold_number}-{index}",
                    "source_row_id": funding_id,
                    "source_artifact_sha256": funding_artifact,
                    "boundary": stamp(when),
                    "observed_rate": rate,
                    "signed_open_notional": position * prior,
                    "signed_cashflow": funding_cash,
                }
            )
        # The first and final bar of each segment deliberately enter and flatten.
        is_entry = index in entry_indices
        is_flatten = index in flatten_indices
        event_type = (
            "entry"
            if is_entry
            else "liquidation"
            if terminal_liquidation and index == max(flatten_indices)
            else "flatten"
        )
        realized = linear = impact = 0.0
        start_position = position
        start_basis = basis
        active_at_start = active_stop
        prior_equity = cash + position * (prior - basis)
        prior_cash = cash
        adverse_price = low if start_position > 0 else high
        worst_intrabar_equity = (
            prior_cash + funding_cash + start_position * (adverse_price - start_basis)
        )
        maintenance_margin_required = abs(start_position * adverse_price) * 0.0055
        if is_entry or is_flatten:
            qty = (trade_quantities or {}).get(index, trade_quantity) if is_entry else -position
            stop_id = f"stop-{fold_number}-{index}" if is_entry else active_stop
            order_id = f"o-{fold_number}-{index}"
            fill_id = f"x-{fold_number}-{index}"
            common = {
                "sequence_id": sequence,
                "fold_id": fold_name,
                "variant_id": candidate_id,
                "leaf_id": leaf,
                "engine_receipt_sha256": receipt_hash,
                "cost_bps": cost_bps,
                "period_id": period_id,
                "timestamp": stamp(when),
                "symbol": symbol,
            }
            orders.append(
                {
                    **common,
                    "order_id": order_id,
                    "signed_qty": qty,
                    "signed_quote_notional": qty * mark,
                    "requested_qty": abs(qty),
                    "direction": "BUY" if qty > 0 else "SELL",
                    "order_type": "MKT",
                    "time_in_force": "GTC",
                    "is_maker": False,
                    "is_entry": is_entry,
                    "protective_stop_id": stop_id,
                }
            )
            rate = 0.10 * (abs(qty * mark) / (bar_volume_base * mark)) ** 0.5
            impact = abs(qty * mark) * rate
            linear = abs(qty * mark) * cost_bps / 10_000
            fills.append(
                {
                    **common,
                    "fill_id": fill_id,
                    "order_id": order_id,
                    "is_entry": is_entry,
                    "signed_qty": qty,
                    "requested_qty": abs(qty),
                    "direction": "BUY" if qty > 0 else "SELL",
                    "fill_price": mark,
                    "signed_quote_notional": qty * mark,
                    "is_maker": False,
                    "bar_volume": bar_volume_base,
                    "observed_adv_quote": bar_volume_base * mark,
                    "participation": abs(qty * mark) / (bar_volume_base * mark),
                    "impact_coefficient": 0.10,
                    "sqrt_impact_rate": rate,
                    "sqrt_impact_cash_cost": impact,
                    "protective_stop_id": stop_id,
                    "protective_stop_source": stop_source if is_entry else None,
                }
            )
            events.append(
                {
                    **common,
                    "event_id": f"e-{fold_number}-{index}",
                    "event_index": 0,
                    "fill_id": fill_id,
                    "event_type": event_type,
                }
            )
            if is_entry:
                basis, position, active_stop = mark, qty, stop_id
                stops.append(
                    {
                        "stop_id": stop_id,
                        "symbol": symbol,
                        "entry_fill_id": fill_id,
                        "side": "SELL",
                        "quantity": qty,
                        "stop_price": (
                            stop_price if stop_price is not None else float(int(mark * 0.99))
                        ),
                        "source": stop_source,
                        "activated_period_id": period_id,
                        "deactivated_period_id": None,
                        "trigger_fill_id": None,
                    }
                )
            else:
                realized = position * (mark - basis)
                position, basis = 0.0, 0.0
                assert active_stop is not None
                stops[-1]["deactivated_period_id"] = period_id
                active_stop = None
            cash += funding_cash + realized - linear - impact
        else:
            cash += funding_cash
        if is_entry or is_flatten:
            adverse_price = low if position > 0 else high
            post_fill_adverse_equity = cash + position * (adverse_price - basis)
            if post_fill_adverse_equity <= worst_intrabar_equity:
                worst_intrabar_equity = post_fill_adverse_equity
                maintenance_margin_required = abs(position * adverse_price) * 0.0055
        unrealized = position * (mark - basis)
        gross = realized + unrealized - (prior_equity - prior_cash)
        equity = cash + unrealized
        exposure = (
            max(abs(start_position * prior), abs(start_position * mark), abs(position * mark))
            / prior_equity
        )
        active_ids = (
            [active_stop]
            if active_stop is not None
            else ([active_at_start] if active_at_start is not None else [])
        )
        periods.append(
            {
                "period_id": period_id,
                "timestamp": stamp(when),
                "segment": segment,
                "expected_funding": expected_funding,
                "gross_pnl": gross,
                "linear_cost": linear,
                "impact_cost": impact,
                "funding_cashflow": funding_cash,
                "net_pnl": equity - prior_equity,
                "prior_equity": prior_equity,
                "equity": equity,
                "prior_cash_balance": prior_cash,
                "cash_balance": cash,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "inventory_cost_basis": (
                    []
                    if not position
                    else [{"symbol": symbol, "quantity": position, "average_entry_price": basis}]
                ),
                "gross_exposure_fraction": exposure,
                "raw_net_return": (equity - prior_equity) / prior_equity,
                "exposure_normalized_net_return": (
                    ((equity - prior_equity) / prior_equity) / exposure if exposure else 0.0
                ),
                "position_notional": position * mark,
                "active_protective_stop_ids": active_ids,
                "worst_intrabar_equity": worst_intrabar_equity,
                "maintenance_margin_required": maintenance_margin_required,
            }
        )
        signal["position"] = position
    fold = {
        "fold_id": fold_name,
        "router_execution_receipts_sha256": router_execution_receipts_sha256
        or _canonical_sha256([{"evaluation_mode": "handler"}]),
        "bar_interval_seconds": bar_interval_hours * 3_600,
        "evaluated_range": {"start": stamp(times[0]), "end": stamp(times[14])},
        "validation_range": {"start": stamp(times[0]), "end": stamp(times[4])},
        "locked_oos_range": {"start": stamp(times[6]), "end": stamp(times[14])},
        "purge": {
            "expected_count": 1,
            "removed_range": {"start": stamp(times[4]), "end": stamp(times[5])},
            "removed_rows": [{"timestamp": stamp(times[4])}],
        },
        "embargo": {
            "expected_count": 1,
            "removed_range": {"start": stamp(times[5]), "end": stamp(times[6])},
            "removed_rows": [{"timestamp": stamp(times[5])}],
        },
        "initial_equity": 100_000.0,
        "periods": periods,
        "funding": funding,
        "protective_stops": stops,
        "entry_count": 2,
        "protective_stop_count": 2,
        "liquidation_count": int(terminal_liquidation),
        "ruin": any(period["worst_intrabar_equity"] <= 0 for period in periods),
        "equity": cash,
    }
    return fold, signals, orders, fills, events


def _economic_evidence_and_bindings(
    *, terminal_liquidation: bool = False
) -> tuple[dict[str, Any], cost_proof.ExternalBindings]:
    profile, market_rows, funding_rows = _cost_profile(), {}, {}
    router_hash, membership_hash, receipt = "3" * 64, "4" * 64, "5" * 64
    candidate_scenarios: list[dict[str, Any]] = []
    all_router_tapes: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = {}
    trial_vectors: dict[str, tuple[list[str], list[str], list[float], list[float]]] = {}
    for candidate_id in cost_proof.CANDIDATES:
        scenarios = []
        for bps in cost_proof.COST_LADDER:
            folds, signals, orders, fills, events = [], [], [], [], []
            for number in range(5):
                built = _cost_fold(
                    candidate_id,
                    number,
                    bps,
                    datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=4 * number),
                    market_rows,
                    funding_rows,
                    receipt,
                    prices_override=(
                        [30.0, 30.0, 30.0, 360.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 25.0]
                        if terminal_liquidation
                        else None
                    ),
                    trade_quantities={0: 1.0, 4: 19_400.0} if terminal_liquidation else None,
                    bar_volume_base=10**16 if terminal_liquidation else 100.0,
                    terminal_liquidation=terminal_liquidation,
                    stop_source="strategy" if terminal_liquidation else "engine_default",
                    stop_price=1.0 if terminal_liquidation else None,
                )
                fold, *tapes = built
                folds.append(fold)
                signals.extend(tapes[0])
                orders.extend(tapes[1])
                fills.extend(tapes[2])
                events.extend(tapes[3])
                all_router_tapes[(candidate_id, fold["fold_id"], f"leaf-{number}", bps)] = {
                    name: {
                        "leaf_id": f"leaf-{number}",
                        "engine_receipt_sha256": receipt,
                        "sequence": [row["sequence_id"] for row in rows],
                        "rows": _row_hashes(rows),
                    }
                    for name, rows in (
                        ("signal_position", tapes[0]),
                        ("order", tapes[1]),
                        ("fill", tapes[2]),
                        ("event", tapes[3]),
                    )
                }
            scenario = {
                "cost_bps": bps,
                "evaluation_modes": ["handler"],
                "generic_fallback_proxy_count": 0,
                "current_fold_oos_input_count": 0,
                "router_replay_manifest_sha256": router_hash,
                "membership_sha256": membership_hash,
                "signal_position_tape": signals,
                "orders": orders,
                "fills": fills,
                "events": events,
                "signal_tape_sha256": _canonical_sha256(signals),
                "order_tape_sha256": _canonical_sha256(orders),
                "execution_tape_sha256": _canonical_sha256(fills),
                "event_tape_sha256": _canonical_sha256(events),
                "economic_tape_sha256": _canonical_sha256(cost_proof._economic_tape(folds)),
                "folds": folds,
            }
            scenarios.append(scenario)
            if bps == 20:
                validation = [
                    row["raw_net_return"]
                    for fold in folds
                    for row in fold["periods"]
                    if row["segment"] == "validation"
                ]
                locked = [
                    row["exposure_normalized_net_return"]
                    for fold in folds
                    for row in fold["periods"]
                    if row["segment"] == "locked_oos"
                ]
                validation_ids = [
                    row["period_id"]
                    for fold in folds
                    for row in fold["periods"]
                    if row["segment"] == "validation"
                ]
                locked_ids = [
                    row["period_id"]
                    for fold in folds
                    for row in fold["periods"]
                    if row["segment"] == "locked_oos"
                ]
                trial_vectors[candidate_id] = validation_ids, locked_ids, validation, locked
        candidate_scenarios.append(
            {
                "candidate_id": candidate_id,
                "router_replay_manifest_sha256": router_hash,
                "membership_sha256": membership_hash,
                "scenarios": scenarios,
            }
        )
    provenance = {
        field: _canonical_sha256(field) for field in cost_proof.PROVENANCE_ARTIFACTS.values()
    }
    provenance["router_replay_manifest_sha256"], provenance["membership_sha256"] = (
        router_hash,
        membership_hash,
    )
    provenance["candidate_ids_sha256"] = cost_proof.candidate_ids_sha256()
    evidence = {
        "schema": cost_proof.SCHEMA,
        "candidate_ids": list(cost_proof.CANDIDATES),
        "cost_ladder_bps": list(cost_proof.COST_LADDER),
        "cscv_splits": cost_proof.CSCV_SPLITS,
        "provenance": provenance,
        "candidates": candidate_scenarios,
    }
    validation_ids, locked_ids, _, _ = trial_vectors[cost_proof.CANDIDATES[0]]
    trial_result_artifacts, trials = {}, []
    for ordinal, candidate_id in enumerate(cost_proof.CANDIDATES):
        validation_ids, locked_ids, validation, locked = trial_vectors[candidate_id]
        digest = f"{ordinal + 1:x}" * 64
        result = {
            "schema": "cost_proof_trial_result_v2",
            "trial_id": candidate_id,
            "ordinal": ordinal,
            "registered_at_utc": f"2025-02-0{ordinal + 1}T00:00:00Z",
            "completed_at_utc": f"2025-02-0{ordinal + 1}T01:00:00Z",
            "status": "succeeded",
            "status_reason": None,
            "dedup_representative_trial_id": candidate_id,
            "validation_period_ids": validation_ids,
            "locked_oos_period_ids": locked_ids,
            "validation_normalized_returns": validation,
            "locked_oos_normalized_returns": locked,
        }
        trials.append(
            {
                key: result[key]
                for key in (
                    "trial_id",
                    "ordinal",
                    "registered_at_utc",
                    "completed_at_utc",
                    "status",
                    "status_reason",
                    "dedup_representative_trial_id",
                )
            }
            | {"result_artifact_sha256": digest}
        )
        trial_result_artifacts[digest] = result
    for status, reason in (("failed", "engine failed"), ("skipped", "duplicate")):
        ordinal = len(trials)
        digest = f"{ordinal + 1:x}" * 64
        representative = None if status == "failed" else cost_proof.CANDIDATES[0]
        trial = {
            "trial_id": f"{status}-trial",
            "ordinal": ordinal,
            "registered_at_utc": f"2025-02-0{ordinal + 1}T00:00:00Z",
            "completed_at_utc": f"2025-02-0{ordinal + 1}T01:00:00Z",
            "status": status,
            "status_reason": reason,
            "dedup_representative_trial_id": representative,
        }
        trials.append(trial | {"result_artifact_sha256": digest})
        trial_result_artifacts[digest] = {
            "schema": "cost_proof_trial_result_v2",
            **trial,
            "validation_period_ids": validation_ids,
            "locked_oos_period_ids": locked_ids,
            "validation_normalized_returns": [],
            "locked_oos_normalized_returns": [],
        }
    ledger = {
        "schema": "cost_proof_trial_ledger_v2",
        "cost_bps": 20,
        "trials": trials,
        "raw_trial_count": len(trials),
        "effective_trial_count": 2,
        "validation_period_ids": validation_ids,
        "locked_oos_period_ids": locked_ids,
        "validation_period_ids_sha256": _canonical_sha256(validation_ids),
        "locked_oos_period_ids_sha256": _canonical_sha256(locked_ids),
        "trial_projection_sha256": _canonical_sha256(trials),
        "current_fold_oos_input_count": 0,
    }
    router_manifest = {
        "folds": [
            {
                "fold_id": f"fold-{number}",
                "selection": {"leaves": [{"traded_symbols": ["BTC-USD"]}]},
                "variants": [
                    {
                        "variant_id": candidate_id,
                        "execution_receipts": [{"evaluation_mode": "handler"}],
                    }
                    for candidate_id in cost_proof.CANDIDATES
                ],
                "locked_oos": {
                    "start_utc": (
                        datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=4 * number, hours=36)
                    )
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "end_utc": (
                        datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=4 * number, hours=84)
                    )
                    .isoformat()
                    .replace("+00:00", "Z"),
                },
            }
            for number in range(5)
        ]
    }
    hashes = {
        artifact: provenance[field] for artifact, field in cost_proof.PROVENANCE_ARTIFACTS.items()
    }
    bindings = cost_proof.ExternalBindings(
        hashes=hashes,
        profile=profile,
        source_manifest={},
        source_run_receipt={},
        search_run_receipt={"frozen_at_utc": "2025-03-01T00:00:00Z"},
        cost_commit={},
        router_manifest=router_manifest,
        membership={},
        trial_ledger=ledger,
        trial_result_artifacts=trial_result_artifacts,
        market_artifact_hashes=frozenset({"1" * 64, "6" * 64}),
        funding_artifact_hashes=frozenset({"2" * 64}),
        market_rows=market_rows,
        funding_rows=funding_rows,
        router_tapes=all_router_tapes,
        trusted_roots=ROOTS,
    )
    return evidence, bindings


def test_cost_proof_v2_complete_economic_bundle_passes() -> None:
    evidence, bindings = _economic_evidence_and_bindings()
    report = cost_proof.evaluate_cost_proof(evidence, bindings=bindings)
    assert report.status == "PASS"
    assert report.selected_candidate_id == cost_proof.CANDIDATES[0]


def test_cost_proof_v2_authenticated_immediate_breach_liquidation_rejects_economically() -> None:
    evidence, bindings = _economic_evidence_and_bindings(terminal_liquidation=True)

    report = cost_proof.evaluate_cost_proof(evidence, bindings=bindings)

    assert report.status == "REJECT"
    assert report.reasons == ("no candidate passed",)
    assert all(
        "liquidation or ruin" in candidate["reasons"] for candidate in report.candidate_reports
    )


def _append_residual_breach_action(
    evidence: dict[str, Any], bindings: cost_proof.ExternalBindings
) -> cost_proof.ExternalBindings:
    candidate = evidence["candidates"][0]
    for scenario_index, scenario in enumerate(candidate["scenarios"]):
        fold = scenario["folds"][0]
        liquidation = next(
            row
            for row in scenario["events"]
            if row["fold_id"] == fold["fold_id"] and row["event_type"] == "liquidation"
        )
        fill = next(row for row in scenario["fills"] if row["fill_id"] == liquidation["fill_id"])
        order = next(row for row in scenario["orders"] if row["order_id"] == fill["order_id"])
        quantity, residual = (
            abs(float(fill["signed_qty"])) * 0.01,
            abs(float(fill["signed_qty"])) * 0.99,
        )
        price, volume = float(fill["fill_price"]), float(fill["bar_volume"])
        for row in (order, fill):
            row["signed_qty"] = -quantity
            row["requested_qty"] = quantity
            row["signed_quote_notional"] = -quantity * price
        fill["participation"] = quantity / volume
        fill["sqrt_impact_rate"] = 0.10 * (quantity / volume) ** 0.5
        fill["sqrt_impact_cash_cost"] = quantity * price * fill["sqrt_impact_rate"]
        fill["protective_stop_id"] = None
        order["protective_stop_id"] = None
        residual_order = deepcopy(order)
        residual_fill = deepcopy(fill)
        residual_event = deepcopy(liquidation)
        residual_order.update(
            {
                "sequence_id": f"{order['sequence_id']}-residual",
                "order_id": f"{order['order_id']}-residual",
                "signed_qty": -residual,
                "requested_qty": residual,
                "signed_quote_notional": -residual * price,
                "protective_stop_id": fold["protective_stops"][-1]["stop_id"],
            }
        )
        residual_fill.update(
            {
                "sequence_id": f"{fill['sequence_id']}-residual",
                "fill_id": f"{fill['fill_id']}-residual",
                "order_id": residual_order["order_id"],
                "signed_qty": -residual,
                "requested_qty": residual,
                "signed_quote_notional": -residual * price,
                "participation": residual / volume,
                "sqrt_impact_rate": 0.10 * (residual / volume) ** 0.5,
                "sqrt_impact_cash_cost": residual * price * (0.10 * (residual / volume) ** 0.5),
                "protective_stop_id": fold["protective_stops"][-1]["stop_id"],
            }
        )
        residual_event.update(
            {
                "sequence_id": f"{liquidation['sequence_id']}-residual",
                "event_id": f"{liquidation['event_id']}-residual",
                "event_index": 1,
                "fill_id": residual_fill["fill_id"],
                "event_type": "flatten",
            }
        )
        scenario["orders"].insert(scenario["orders"].index(order) + 1, residual_order)
        scenario["fills"].insert(scenario["fills"].index(fill) + 1, residual_fill)
        event_index = scenario["events"].index(liquidation)
        scenario["events"].insert(event_index + 1, residual_event)
        bindings = _reroot_scenario_tapes(
            evidence, bindings, candidate_index=0, scenario_index=scenario_index
        )
    return bindings


def test_cost_proof_v2_residual_immediate_breach_requires_next_liquidation() -> None:
    evidence, bindings = _economic_evidence_and_bindings(terminal_liquidation=True)
    bindings = _append_residual_breach_action(evidence, bindings)

    report = cost_proof.evaluate_cost_proof(evidence, bindings=bindings)

    assert report.status == "STOP"
    assert any("engine ledger does not reconcile" in reason for reason in report.reasons)


def _write_canonical(path: Path, value: Any) -> str:
    path.write_bytes(
        json.dumps(
            value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_cost_bundle(
    tmp_path: Path, *, oversized_market: bool = False
) -> tuple[cost_proof.CostProofReport, list[str]]:
    """Exercise the public file API with Router's authenticated cost-tape ownership."""
    router_dir = tmp_path / "router"
    router_dir.mkdir()
    cache: dict[
        tuple[str, str, int],
        tuple[
            dict[str, Any],
            list[dict[str, Any]],
            list[dict[str, Any]],
            list[dict[str, Any]],
            list[dict[str, Any]],
        ],
    ] = {}
    market_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    funding_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    market_token, funding_token = "0" * 64, "9" * 64
    prices = [100.0, 110.0, 120.0, 130.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]
    seed_market: dict[tuple[str, str, str], dict[str, Any]] = {}
    seed_funding: dict[tuple[str, str, str], dict[str, Any]] = {}
    for number in range(5):
        _cost_fold(
            cost_proof.CANDIDATES[0],
            number,
            10,
            datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=number * 3 - 1, hours=6),
            seed_market,
            seed_funding,
            "b" * 64,
            bar_interval_hours=3,
            leaf_id="leaf",
            market_artifact=market_token,
            funding_artifact=funding_token,
            fold_id=f"f{number}",
            prices_override=prices,
            symbol="BTCUSDT",
        )
    market_records = [
        {
            "source_row_id": row_id,
            "symbol": symbol,
            "timestamp": row["timestamp"],
            "prior_mark_price": row["prior_mark_price"],
            "mark_price": row["mark_price"],
            "open": row["prior_mark_price"],
            "high": row["high"],
            "low": row["low"],
            "close": row["mark_price"],
            "bar_volume_base": row["bar_volume_base"],
            "price_tick_size": row["price_tick_size"],
            "quantity_step_size": row["quantity_step_size"],
        }
        for (_artifact, row_id, symbol), row in sorted(
            seed_market.items(), key=lambda item: item[1]["timestamp"]
        )
    ]
    funding_records = [
        {"source_row_id": row_id, "symbol": symbol, **row}
        for (_artifact, row_id, symbol), row in sorted(
            seed_funding.items(), key=lambda item: item[1]["boundary"]
        )
    ]
    if not funding_records:
        funding_records = [
            {
                "source_row_id": "funding-coverage",
                "symbol": "BTCUSDT",
                "boundary": "2024-12-31T08:00:00Z",
                "observed_rate": 0.0001,
            }
        ]
    market_path, funding_path = tmp_path / "market.json", tmp_path / "funding.json"
    market_token = _write_canonical(
        market_path, {"schema": "cost_proof_market_artifact_v1", "rows": market_records}
    )
    funding_token = _write_canonical(
        funding_path, {"schema": "cost_proof_funding_artifact_v1", "rows": funding_records}
    )

    def cost_rows(
        fold_id: str, variant_id: str, leaf_id: str, bps: int, kind: str, engine_receipt: str
    ) -> list[dict[str, Any]]:
        number = int(fold_id[1:])
        key = variant_id, fold_id, bps
        if key not in cache:
            locked_start = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=number * 3)
            cache[key] = _cost_fold(
                variant_id,
                number,
                bps,
                locked_start - timedelta(hours=18),
                market_rows,
                funding_rows,
                engine_receipt,
                bar_interval_hours=3,
                leaf_id=leaf_id,
                market_artifact=market_token,
                funding_artifact=funding_token,
                fold_id=fold_id,
                prices_override=prices,
                trade_quantity=(1.0 if variant_id == cost_proof.CANDIDATES[0] else 4.0),
                symbol="BTCUSDT",
            )
        return {
            "cost_signal_position_tape": cache[key][1],
            "cost_order_tape": cache[key][2],
            "cost_fill_tape": cache[key][3],
            "cost_event_tape": cache[key][4],
        }[kind]

    manifest, router_paths, router_roots = _bundle(
        router_dir, mode="scaled", cost_rows=cost_rows, all_folds_scaled=True
    )
    router_artifacts = {digest: path for digest, path in router_paths.items() if len(digest) == 64}
    receipt_hashes = {
        (fold["fold_id"], variant["variant_id"]): _canonical_sha256(variant["execution_receipts"])
        for fold in manifest["folds"]
        for variant in fold["variants"]
    }
    for (candidate, fold_id, _bps), (fold, signals, orders, fills, events) in cache.items():
        fold["router_execution_receipts_sha256"] = receipt_hashes[fold_id, candidate]
        for rows in (signals, orders, fills, events):
            assert all(row["engine_receipt_sha256"] for row in rows)
    assert len(cache) == 40
    market_records = [
        {
            "source_row_id": row_id,
            "symbol": symbol,
            "timestamp": row["timestamp"],
            "prior_mark_price": row["prior_mark_price"],
            "mark_price": row["mark_price"],
            "open": row["prior_mark_price"],
            "high": row["high"],
            "low": row["low"],
            "close": row["mark_price"],
            "bar_volume_base": row["bar_volume_base"],
            "price_tick_size": row["price_tick_size"],
            "quantity_step_size": row["quantity_step_size"],
        }
        for (_artifact, row_id, symbol), row in sorted(
            market_rows.items(), key=lambda item: item[1]["timestamp"]
        )
    ]
    if oversized_market:
        market_records[0]["mark_price"] = 10**4_000
    actual_funding_records = [
        {"source_row_id": row_id, "symbol": symbol, **row}
        for (_artifact, row_id, symbol), row in sorted(
            funding_rows.items(), key=lambda item: item[1]["boundary"]
        )
    ]
    if actual_funding_records:
        funding_records = actual_funding_records
    market_path, funding_path = tmp_path / "market.json", tmp_path / "funding.json"
    market_digest = _write_canonical(
        market_path, {"schema": "cost_proof_market_artifact_v1", "rows": market_records}
    )
    funding_digest = _write_canonical(
        funding_path, {"schema": "cost_proof_funding_artifact_v1", "rows": funding_records}
    )
    for fold, signals, _orders, _fills, _events in cache.values():
        for signal in signals:
            signal["market_data_artifact_sha256"] = market_digest
        for settlement in fold["funding"]:
            settlement["source_artifact_sha256"] = funding_digest

    candidate_scenarios, trial_vectors = [], {}
    for candidate_id in cost_proof.CANDIDATES:
        scenarios = []
        for bps in cost_proof.COST_LADDER:
            cells = [cache[candidate_id, f"f{number}", bps] for number in range(5)]
            folds = [cell[0] for cell in cells]
            tapes = [[row for cell in cells for row in cell[index]] for index in range(1, 5)]
            scenarios.append(
                {
                    "cost_bps": bps,
                    "evaluation_modes": ["handler"],
                    "generic_fallback_proxy_count": 0,
                    "current_fold_oos_input_count": 0,
                    "router_replay_manifest_sha256": hashlib.sha256(
                        router_paths["manifest"].read_bytes()
                    ).hexdigest(),
                    "membership_sha256": hashlib.sha256(
                        router_paths["membership"].read_bytes()
                    ).hexdigest(),
                    "signal_position_tape": tapes[0],
                    "orders": tapes[1],
                    "fills": tapes[2],
                    "events": tapes[3],
                    "signal_tape_sha256": _canonical_sha256(tapes[0]),
                    "order_tape_sha256": _canonical_sha256(tapes[1]),
                    "execution_tape_sha256": _canonical_sha256(tapes[2]),
                    "event_tape_sha256": _canonical_sha256(tapes[3]),
                    "economic_tape_sha256": _canonical_sha256(cost_proof._economic_tape(folds)),
                    "folds": folds,
                }
            )
            if bps == 20:
                trial_vectors[candidate_id] = (
                    [
                        row["period_id"]
                        for fold in folds
                        for row in fold["periods"]
                        if row["segment"] == "validation"
                    ],
                    [
                        row["period_id"]
                        for fold in folds
                        for row in fold["periods"]
                        if row["segment"] == "locked_oos"
                    ],
                    [
                        row["raw_net_return"]
                        for fold in folds
                        for row in fold["periods"]
                        if row["segment"] == "validation"
                    ],
                    [
                        row["exposure_normalized_net_return"]
                        for fold in folds
                        for row in fold["periods"]
                        if row["segment"] == "locked_oos"
                    ],
                )
        candidate_scenarios.append(
            {
                "candidate_id": candidate_id,
                "router_replay_manifest_sha256": hashlib.sha256(
                    router_paths["manifest"].read_bytes()
                ).hexdigest(),
                "membership_sha256": hashlib.sha256(
                    router_paths["membership"].read_bytes()
                ).hexdigest(),
                "scenarios": scenarios,
            }
        )

    trial_paths, trials = {}, []
    for ordinal, candidate_id in enumerate(cost_proof.CANDIDATES):
        validation_ids, locked_ids, validation, locked = trial_vectors[candidate_id]
        result = {
            "schema": "cost_proof_trial_result_v2",
            "trial_id": candidate_id,
            "ordinal": ordinal,
            "registered_at_utc": f"2025-02-0{ordinal + 1}T00:00:00Z",
            "completed_at_utc": f"2025-02-0{ordinal + 1}T01:00:00Z",
            "status": "succeeded",
            "status_reason": None,
            "dedup_representative_trial_id": candidate_id,
            "validation_period_ids": validation_ids,
            "locked_oos_period_ids": locked_ids,
            "validation_normalized_returns": validation,
            "locked_oos_normalized_returns": locked,
        }
        path = tmp_path / f"trial-{ordinal}.json"
        digest = _write_canonical(path, result)
        trial_paths[digest] = path
        trials.append(
            {
                key: result[key]
                for key in (
                    "trial_id",
                    "ordinal",
                    "registered_at_utc",
                    "completed_at_utc",
                    "status",
                    "status_reason",
                    "dedup_representative_trial_id",
                )
            }
            | {"result_artifact_sha256": digest}
        )
    validation_ids, locked_ids, _, _ = trial_vectors[cost_proof.CANDIDATES[0]]
    ledger = {
        "schema": "cost_proof_trial_ledger_v2",
        "cost_bps": 20,
        "trials": trials,
        "raw_trial_count": 2,
        "effective_trial_count": 2,
        "validation_period_ids": validation_ids,
        "locked_oos_period_ids": locked_ids,
        "validation_period_ids_sha256": _canonical_sha256(validation_ids),
        "locked_oos_period_ids_sha256": _canonical_sha256(locked_ids),
        "trial_projection_sha256": _canonical_sha256(trials),
        "current_fold_oos_input_count": 0,
    }
    ledger_path = tmp_path / "ledger.json"
    _write_canonical(ledger_path, ledger)
    source_manifest = {
        "schema": "cost_proof_source_data_v2",
        "source_run_id": "router-cost-source",
        "synthetic_source_count": 0,
        "actual_funding": True,
        "point_in_time_membership": True,
        "post_append_strict_receipt_sha256": "a" * 64,
        "artifacts": [
            {
                "kind": "market",
                "artifact_sha256": market_digest,
                "row_count": len(market_records),
                "min_timestamp_utc": market_records[0]["timestamp"],
                "max_timestamp_utc": market_records[-1]["timestamp"],
            },
            {
                "kind": "funding",
                "artifact_sha256": funding_digest,
                "row_count": len(funding_records),
                "min_timestamp_utc": funding_records[0]["boundary"],
                "max_timestamp_utc": funding_records[-1]["boundary"],
            },
        ],
    }
    source_path, source_receipt_path = (
        tmp_path / "source-manifest.json",
        tmp_path / "source-receipt.json",
    )
    source_digest = _write_canonical(source_path, source_manifest)
    source_receipt = {
        "schema": "cost_proof_source_run_receipt_v1",
        "source_run_id": "router-cost-source",
        "manifest_sha256": source_digest,
        "artifacts": source_manifest["artifacts"],
        "producer_source_sha256": hashlib.sha256(router_paths["producer"].read_bytes()).hexdigest(),
        "source_commit_sha256": "a" * 64,
        "committed_at_utc": "2025-01-15T00:00:00Z",
    }
    source_receipt_digest = _write_canonical(source_receipt_path, source_receipt)
    profile_path = router_paths["profile"]
    router_manifest_digest = hashlib.sha256(router_paths["manifest"].read_bytes()).hexdigest()
    membership_digest = hashlib.sha256(router_paths["membership"].read_bytes()).hexdigest()
    evidence = {
        "schema": cost_proof.SCHEMA,
        "candidate_ids": list(cost_proof.CANDIDATES),
        "cost_ladder_bps": list(cost_proof.COST_LADDER),
        "cscv_splits": cost_proof.CSCV_SPLITS,
        "provenance": {
            "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            "source_data_manifest_sha256": source_digest,
            "source_run_receipt_sha256": source_receipt_digest,
            "search_run_receipt_sha256": "",
            "router_replay_manifest_sha256": router_manifest_digest,
            "router_source_artifact_sha256": router_roots["source"],
            "lifecycle_sha256": hashlib.sha256(router_paths["lifecycle"].read_bytes()).hexdigest(),
            "membership_sha256": membership_digest,
            "trial_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
            "producer_source_sha256": hashlib.sha256(
                router_paths["producer"].read_bytes()
            ).hexdigest(),
            "router_producer_source_sha256": hashlib.sha256(
                router_paths["producer"].read_bytes()
            ).hexdigest(),
            "router_commit_receipt_sha256": router_roots["commit"],
            "verifier_source_sha256": hashlib.sha256(
                Path(cost_proof.__file__).read_bytes()
            ).hexdigest(),
            "candidate_ids_sha256": cost_proof.candidate_ids_sha256(),
        },
        "candidates": candidate_scenarios,
    }
    search_path = tmp_path / "search.json"
    search_receipt = {
        "schema": "cost_proof_search_run_receipt_v2",
        "trial_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "trial_result_artifacts": [{"artifact_sha256": digest} for digest in trial_paths],
        "candidate_ids": list(cost_proof.CANDIDATES),
        "candidate_ids_sha256": cost_proof.candidate_ids_sha256(),
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": source_digest,
        "router_manifest_sha256": router_manifest_digest,
        "lifecycle_sha256": hashlib.sha256(router_paths["lifecycle"].read_bytes()).hexdigest(),
        "membership_sha256": membership_digest,
        "post_oos_research_variant": True,
        "post_oos_augment": False,
        "post_oos_augmentation_count": 0,
        "current_fold_oos_input_count": 0,
        "new_grid_search": False,
        "recompute_from_json": False,
        "frozen_at_utc": "2025-03-01T00:00:00Z",
        "trial_projection_sha256": ledger["trial_projection_sha256"],
        "validation_period_ids_sha256": ledger["validation_period_ids_sha256"],
        "locked_oos_period_ids_sha256": ledger["locked_oos_period_ids_sha256"],
    }
    search_digest = _write_canonical(search_path, search_receipt)
    evidence["provenance"]["search_run_receipt_sha256"] = search_digest
    evidence_path = tmp_path / "evidence.json"
    evidence_digest = _write_canonical(evidence_path, evidence)
    router_tapes = []
    for digest, path in router_artifacts.items():
        receipt = json.loads(path.read_text())
        if receipt.get("schema") != "router_cost_tape_receipt_v1":
            continue
        for tape in receipt["tapes"]:
            signal = json.loads(router_artifacts[tape["signal_position_sha256"]].read_text())
            order = json.loads(router_artifacts[tape["order_tape_sha256"]].read_text())
            fill = json.loads(router_artifacts[tape["fill_tape_sha256"]].read_text())
            event = json.loads(router_artifacts[tape["event_tape_sha256"]].read_text())
            router_tapes.append(
                {
                    "variant_id": receipt["variant_id"],
                    "fold_id": receipt["fold_id"],
                    "leaf_id": receipt["leaf_id"],
                    "cost_bps": tape["cost_bps"],
                    "receipt_sha256": _canonical_sha256(receipt),
                    "signal_position_sha256": _canonical_sha256(signal),
                    "order_sha256": _canonical_sha256(order),
                    "fill_sha256": _canonical_sha256(fill),
                    "event_sha256": _canonical_sha256(event),
                }
            )
    router_tapes.sort(
        key=lambda row: (row["variant_id"], row["fold_id"], row["leaf_id"], row["cost_bps"])
    )
    commit_path = tmp_path / "cost-commit.json"
    cost_commit = {
        "schema": "cost_proof_commit_v2",
        "evidence_sha256": evidence_digest,
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": source_digest,
        "source_run_receipt_sha256": source_receipt_digest,
        "search_run_receipt_sha256": search_digest,
        "trial_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "router_manifest_sha256": router_manifest_digest,
        "lifecycle_sha256": hashlib.sha256(router_paths["lifecycle"].read_bytes()).hexdigest(),
        "membership_sha256": membership_digest,
        "producer_source_sha256": hashlib.sha256(router_paths["producer"].read_bytes()).hexdigest(),
        "verifier_source_sha256": hashlib.sha256(
            Path(cost_proof.__file__).read_bytes()
        ).hexdigest(),
        "candidate_ids": list(cost_proof.CANDIDATES),
        "candidate_ids_sha256": cost_proof.candidate_ids_sha256(),
        "source_artifacts": source_manifest["artifacts"],
        "trial_result_artifacts": [{"artifact_sha256": digest} for digest in trial_paths],
        "router_tapes": router_tapes,
        "trial_projection_sha256": ledger["trial_projection_sha256"],
        "validation_period_ids_sha256": ledger["validation_period_ids_sha256"],
        "locked_oos_period_ids_sha256": ledger["locked_oos_period_ids_sha256"],
        "committed_at_utc": "2025-03-02T00:00:00Z",
    }
    cost_commit_digest = _write_canonical(commit_path, cost_commit)
    report = cost_proof.evaluate_cost_proof_file(
        evidence_path,
        profile_path,
        source_data_manifest_path=source_path,
        source_run_receipt_path=source_receipt_path,
        search_run_receipt_path=search_path,
        router_replay_manifest_path=router_paths["manifest"],
        router_source_artifact_path=router_paths["source"],
        lifecycle_path=router_paths["lifecycle"],
        membership_path=router_paths["membership"],
        trial_ledger_path=ledger_path,
        producer_source_path=router_paths["producer"],
        commit_receipt_path=commit_path,
        router_producer_source_path=router_paths["producer"],
        router_commit_receipt_path=router_paths["commit"],
        market_artifact_paths={market_digest: market_path},
        funding_artifact_paths={funding_digest: funding_path},
        router_artifact_paths=router_artifacts,
        trial_result_artifact_paths=trial_paths,
        trusted_roots={
            "source_data_commit_sha256": source_receipt_digest,
            "search_run_receipt_sha256": search_digest,
            "cost_proof_commit_sha256": cost_commit_digest,
            "router_source_artifact_sha256": router_roots["source"],
            "router_commit_receipt_sha256": router_roots["commit"],
        },
    )
    argv = [
        "cost-proof",
        "--input",
        str(evidence_path),
        "--config",
        str(profile_path),
        "--source-data-manifest",
        str(source_path),
        "--source-run-receipt",
        str(source_receipt_path),
        "--search-run-receipt",
        str(search_path),
        "--router-replay-manifest",
        str(router_paths["manifest"]),
        "--router-source-artifact",
        str(router_paths["source"]),
        "--lifecycle",
        str(router_paths["lifecycle"]),
        "--membership",
        str(router_paths["membership"]),
        "--trial-ledger",
        str(ledger_path),
        "--producer-source",
        str(router_paths["producer"]),
        "--commit-receipt",
        str(commit_path),
        "--router-producer-source",
        str(router_paths["producer"]),
        "--router-commit-receipt",
        str(router_paths["commit"]),
        "--source-data-commit-sha256",
        source_receipt_digest,
        "--search-run-receipt-sha256",
        search_digest,
        "--cost-proof-commit-sha256",
        cost_commit_digest,
        "--router-source-artifact-sha256",
        router_roots["source"],
        "--router-commit-receipt-sha256",
        router_roots["commit"],
    ]
    for digest, path in {market_digest: market_path}.items():
        argv.extend(("--market-artifact", f"{digest}={path}"))
    for digest, path in {funding_digest: funding_path}.items():
        argv.extend(("--funding-artifact", f"{digest}={path}"))
    for digest, path in router_artifacts.items():
        argv.extend(("--router-artifact", f"{digest}={path}"))
    for digest, path in trial_paths.items():
        argv.extend(("--trial-result-artifact", f"{digest}={path}"))
    return report, argv


def test_cost_proof_v2_public_file_boundary_passes_with_router_bundle(tmp_path: Path) -> None:
    report, _ = _public_cost_bundle(tmp_path)

    assert report.status == "PASS", report.to_json()
    assert report.selected_candidate_id == cost_proof.CANDIDATES[0]


def test_cost_proof_v2_public_file_boundary_oversized_market_stops(
    tmp_path: Path,
) -> None:
    report, argv = _public_cost_bundle(tmp_path, oversized_market=True)

    assert report.status == "STOP"
    assert cli_main(argv) == 2


def _mutated_cost_proof(
    mutation: str, value: Any = None
) -> tuple[dict[str, Any], cost_proof.ExternalBindings]:
    evidence, bindings = _economic_evidence_and_bindings()
    evidence = deepcopy(evidence)
    candidate = evidence["candidates"][0]
    scenario = candidate["scenarios"][0]
    fold = scenario["folds"][0]
    if mutation == "cost_bps":
        scenario["cost_bps"] = value
    elif mutation == "cross_rung_tape":
        scenario["signal_position_tape"] = deepcopy(
            candidate["scenarios"][1]["signal_position_tape"]
        )
        scenario["signal_tape_sha256"] = _canonical_sha256(scenario["signal_position_tape"])
    elif mutation == "router_row":
        tapes = deepcopy(bindings.router_tapes)
        key = next(iter(tapes))
        tapes[key]["signal_position"]["rows"][0]["row_sha256"] = "0" * 64
        bindings = replace(bindings, router_tapes=tapes)
    elif mutation == "funding":
        fold["funding"] = fold["funding"][:-1] if value == "missing" else fold["funding"] * 2
    elif mutation == "stop":
        if value == "missing":
            fold["protective_stops"] = []
        elif value == "pre_entry":
            fold["protective_stops"][0]["activated_period_id"] = fold["periods"][1]["period_id"]
        else:
            fold["protective_stops"][0]["deactivated_period_id"] = fold["periods"][2]["period_id"]
    elif mutation == "order_type":
        scenario["orders"][0]["order_type"] = "LIMIT"
    elif mutation == "accounting":
        fold["periods"][0][value] += 1.0
    elif mutation == "oos":
        scenario["current_fold_oos_input_count"] = 1
    elif mutation == "fallback":
        scenario["generic_fallback_proxy_count"] = 1
    elif mutation == "provenance":
        evidence["provenance"][value] = "0" * 64
    return evidence, bindings


@pytest.mark.parametrize(
    ("mutation", "value", "reason"),
    [
        ("cost_bps", "10", "scenario order/count mismatch"),
        ("cost_bps", True, "scenario order/count mismatch"),
        ("cost_bps", 10.0, "scenario order/count mismatch"),
        ("cross_rung_tape", None, "engine ledger does not reconcile"),
        ("router_row", None, "engine ledger does not reconcile"),
        ("funding", "missing", "engine ledger does not reconcile"),
        ("funding", "duplicate", "engine ledger does not reconcile"),
        ("stop", "missing", "engine ledger does not reconcile"),
        ("stop", "pre_entry", "engine ledger does not reconcile"),
        ("stop", "after_flatten", "engine ledger does not reconcile"),
        ("order_type", None, "engine ledger does not reconcile"),
        ("accounting", "cash_balance", "engine ledger does not reconcile"),
        ("accounting", "equity", "engine ledger does not reconcile"),
        ("accounting", "worst_intrabar_equity", "engine ledger does not reconcile"),
        ("accounting", "maintenance_margin_required", "engine ledger does not reconcile"),
        ("accounting", "gross_exposure_fraction", "engine ledger does not reconcile"),
        ("oos", None, "unsafe evaluation evidence"),
        ("fallback", None, "unsafe evaluation evidence"),
        ("provenance", "source_run_receipt_sha256", "source_run_receipt SHA mismatch"),
        ("provenance", "search_run_receipt_sha256", "search_run_receipt SHA mismatch"),
        ("provenance", "router_replay_manifest_sha256", "router_replay_manifest SHA mismatch"),
        ("provenance", "trial_ledger_sha256", "trial_ledger SHA mismatch"),
    ],
    ids=[
        "string-cost",
        "bool-cost",
        "coerced-cost",
        "cross-rung-tape",
        "router-row-ownership",
        "missing-funding",
        "duplicate-funding",
        "missing-stop",
        "pre-entry-stop",
        "after-flatten-stop",
        "unknown-order-type",
        "cash",
        "equity",
        "worst-equity",
        "maintenance",
        "gross-exposure",
        "current-fold-oos",
        "generic-fallback",
        "source-provenance",
        "search-provenance",
        "router-provenance",
        "trial-provenance",
    ],
)
def test_cost_proof_v2_adversarial_contract_exploit_matrix(
    mutation: str, value: Any, reason: str
) -> None:
    evidence, bindings = _mutated_cost_proof(mutation, value)
    report = cost_proof.evaluate_cost_proof(evidence, bindings=bindings)
    assert report.status == "STOP"
    assert any(reason in item for item in report.reasons)


def test_cost_proof_v2_rejects_understated_pre_fill_exposure() -> None:
    evidence, bindings = _economic_evidence_and_bindings()
    fold = evidence["candidates"][0]["scenarios"][0]["folds"][0]
    flattened = fold["periods"][3]
    assert flattened["position_notional"] == 0.0
    assert flattened["gross_exposure_fraction"] > 0.0
    flattened["gross_exposure_fraction"] = 0.0

    report = cost_proof.evaluate_cost_proof(evidence, bindings=bindings)

    assert report.status == "STOP"
    assert any("engine ledger does not reconcile" in reason for reason in report.reasons)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("cost_ladder_item", True),
        ("cost_ladder_item", 10.0),
        ("cscv_splits", True),
        ("cscv_splits", float(cost_proof.CSCV_SPLITS)),
        ("scenario_cost_bps", True),
        ("scenario_cost_bps", 10.0),
        ("ledger_cost_bps", True),
        ("ledger_cost_bps", 20.0),
        ("raw_trial_count", True),
        ("raw_trial_count", 4.0),
        ("effective_trial_count", True),
        ("effective_trial_count", 2.0),
        ("ledger_current_fold_oos", True),
        ("ledger_current_fold_oos", 0.0),
        ("trial_ordinal", True),
        ("trial_ordinal", 0.0),
        ("result_ordinal", True),
        ("result_ordinal", 0.0),
    ],
)
def test_cost_proof_v2_rejects_non_native_integer_certification_fields(
    target: str, value: object
) -> None:
    evidence, bindings = _economic_evidence_and_bindings()
    if target == "cost_ladder_item":
        evidence["cost_ladder_bps"][0] = value
    elif target == "cscv_splits":
        evidence[target] = value
    elif target == "scenario_cost_bps":
        evidence["candidates"][0]["scenarios"][0]["cost_bps"] = value
    else:
        ledger = deepcopy(bindings.trial_ledger)
        artifacts = deepcopy(bindings.trial_result_artifacts)
        if target == "ledger_cost_bps":
            ledger["cost_bps"] = value
        elif target in {"raw_trial_count", "effective_trial_count"}:
            ledger[target] = value
        elif target == "ledger_current_fold_oos":
            ledger["current_fold_oos_input_count"] = value
        else:
            digest = ledger["trials"][0]["result_artifact_sha256"]
            if target == "trial_ordinal":
                ledger["trials"][0]["ordinal"] = value
            else:
                artifacts[digest]["ordinal"] = value
            ledger["trial_projection_sha256"] = _canonical_sha256(ledger["trials"])
        bindings = replace(bindings, trial_ledger=ledger, trial_result_artifacts=artifacts)

    report = cost_proof.evaluate_cost_proof(evidence, bindings=bindings)

    assert report.status == "STOP"


def test_whole_family_spa_pvalues_are_finite_corrected_and_repeatable() -> None:
    matrix = np.asarray(
        [
            np.linspace(0.001, 0.016, 16),
            np.linspace(-0.016, -0.001, 16),
            np.ones(16),
        ]
    )

    first = cost_proof._whole_family_spa_pvalues(matrix)
    second = cost_proof._whole_family_spa_pvalues(matrix)

    assert np.array_equal(first, second)
    assert 0.0 < first[0] <= 1.0
    assert first[1] == 1.0
    assert first[2] == 1.0


def test_whole_family_spa_pvalues_use_shared_max_statistic() -> None:
    target = np.linspace(0.001, 0.016, 16)
    alone = cost_proof._whole_family_spa_pvalues(target[None, :])[0]
    family = cost_proof._whole_family_spa_pvalues(
        np.vstack((target, np.linspace(0.016, 0.001, 16)))
    )[0]

    assert family >= alone
    assert family != alone


def test_whole_family_spa_positive_degenerate_resamples_are_conservative() -> None:
    repeated_positive = np.asarray([0.01] * 15 + [-0.01])
    pvalue = cost_proof._whole_family_spa_pvalues(repeated_positive[None, :])[0]

    assert pvalue == pytest.approx(700.0 / 2_001.0)


def _reroot_scenario_tapes(
    evidence: dict[str, Any],
    bindings: cost_proof.ExternalBindings,
    *,
    candidate_index: int,
    scenario_index: int,
) -> cost_proof.ExternalBindings:
    scenario = evidence["candidates"][candidate_index]["scenarios"][scenario_index]
    tape_names = {
        "signal_position_tape": "signal_tape_sha256",
        "orders": "order_tape_sha256",
        "fills": "execution_tape_sha256",
        "events": "event_tape_sha256",
    }
    for tape_name, digest_name in tape_names.items():
        scenario[digest_name] = _canonical_sha256(scenario[tape_name])
    scenario["economic_tape_sha256"] = _canonical_sha256(
        cost_proof._economic_tape(scenario["folds"])
    )

    tapes = deepcopy(bindings.router_tapes)
    candidate_id = evidence["candidates"][candidate_index]["candidate_id"]
    for fold in scenario["folds"]:
        key = (
            candidate_id,
            fold["fold_id"],
            f"leaf-{fold['fold_id'].split('-')[-1]}",
            scenario["cost_bps"],
        )
        for tape_name, router_name in (
            ("signal_position_tape", "signal_position"),
            ("orders", "order"),
            ("fills", "fill"),
            ("events", "event"),
        ):
            rows = [row for row in scenario[tape_name] if row["fold_id"] == fold["fold_id"]]
            tapes[key][router_name]["sequence"] = [row["sequence_id"] for row in rows]
            tapes[key][router_name]["rows"] = _row_hashes(rows)
    return replace(bindings, router_tapes=tapes)


def test_cost_proof_v2_oversized_authenticated_market_number_stops() -> None:
    evidence, bindings = _economic_evidence_and_bindings()
    scenario = evidence["candidates"][0]["scenarios"][0]
    signal = scenario["signal_position_tape"][0]
    signal["mark_price"] = 10**4_000
    market_rows = deepcopy(bindings.market_rows)
    market_rows[
        (
            signal["market_data_artifact_sha256"],
            signal["market_source_row_id"],
            signal["symbol"],
        )
    ]["mark_price"] = signal["mark_price"]
    bindings = replace(
        _reroot_scenario_tapes(evidence, bindings, candidate_index=0, scenario_index=0),
        market_rows=market_rows,
    )

    assert cost_proof.evaluate_cost_proof(evidence, bindings=bindings).status == "STOP"


def test_cost_proof_v2_oversized_authenticated_funding_number_stops() -> None:
    evidence, bindings = _economic_evidence_and_bindings()
    scenario = evidence["candidates"][0]["scenarios"][0]
    funding = scenario["folds"][0]["funding"][0]
    funding["observed_rate"] = 10**4_000
    funding_rows = deepcopy(bindings.funding_rows)
    funding_rows[
        (
            funding["source_artifact_sha256"],
            funding["source_row_id"],
            funding["symbol"],
        )
    ]["observed_rate"] = funding["observed_rate"]
    bindings = replace(
        _reroot_scenario_tapes(evidence, bindings, candidate_index=0, scenario_index=0),
        funding_rows=funding_rows,
    )

    assert cost_proof.evaluate_cost_proof(evidence, bindings=bindings).status == "STOP"


def _healthy_liquidation_stop_report() -> cost_proof.CostProofReport:
    evidence, bindings = _economic_evidence_and_bindings()
    for scenario_index, scenario in enumerate(evidence["candidates"][0]["scenarios"]):
        fold = scenario["folds"][0]
        event = scenario["events"][1]
        assert event["event_type"] == "flatten"
        event["event_type"] = "liquidation"
        fold["liquidation_count"] = 1
        bindings = _reroot_scenario_tapes(
            evidence, bindings, candidate_index=0, scenario_index=scenario_index
        )
    return cost_proof.evaluate_cost_proof(evidence, bindings=bindings)


def test_cost_proof_v2_healthy_immediate_prestate_liquidation_stops() -> None:
    report = _healthy_liquidation_stop_report()

    assert report.status == "STOP"
    assert any("engine ledger does not reconcile" in reason for reason in report.reasons)
