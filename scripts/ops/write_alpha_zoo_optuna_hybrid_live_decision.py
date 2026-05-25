#!/usr/bin/env python3
"""Write a paper/testnet-only live decision for the Alpha Zoo Optuna hybrid."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.strategies.alpha_zoo_optuna_hybrid_live import (
    AlphaZooOptunaHybridLiveConfig,
    DEFAULT_INTEGER_PORTFOLIO_ARTIFACT,
    DEFAULT_OPTUNA_HYBRID_ARTIFACT,
    DEFAULT_SELECTED_PROFILE_ID,
    ROUND_TRIP_COST_BPS,
    RETURN_PER_TURNOVER_THRESHOLD_BPS,
    SourceSleeve,
    load_alpha_zoo_optuna_hybrid_live_config,
)

DEFAULT_OUTPUT_DIR = (
    Path("var")
    / "reports"
    / "profit_moonshot_20260501"
    / "current_tail_20260508"
    / "alpha_v2"
    / "alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524"
)
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "paper_testnet_live_decision_latest.json"


def _target_notional_fraction(
    config: AlphaZooOptunaHybridLiveConfig,
    sleeve: SourceSleeve,
) -> float:
    total = 0.0
    for profile in config.source_profiles:
        if sleeve.model_id not in profile.selected_model_ids:
            continue
        total += float(config.final_profile_weights.get(profile.profile_id, 0.0)) * float(
            profile.leverage_map[sleeve.symbol]
        )
    return float(sleeve.allocation_fraction) * total


def _notional_risk_caps(config: AlphaZooOptunaHybridLiveConfig) -> dict[str, float]:
    target_by_model = {
        sleeve.model_id: _target_notional_fraction(config, sleeve)
        for sleeve in config.source_sleeves
    }
    target_by_symbol: dict[str, float] = {}
    for sleeve in config.source_sleeves:
        target_by_symbol[sleeve.symbol] = target_by_symbol.get(sleeve.symbol, 0.0) + target_by_model[
            sleeve.model_id
        ]
    max_order_target = max(target_by_model.values())
    max_symbol_target = max(target_by_symbol.values())
    gross_target = sum(target_by_model.values())
    return {
        "max_order_value": 0.0,
        "max_order_notional_pct": round(max(max_order_target + 0.05, max_order_target * 1.05), 6),
        "max_symbol_exposure_pct": round(max(max_symbol_target + 0.05, max_symbol_target * 1.05), 6),
        "max_total_margin_pct": round(max(gross_target + 0.25, gross_target * 1.10), 6),
        "max_total_notional_pct": round(max(gross_target + 0.25, gross_target * 1.10), 6),
    }


def build_decision_payload(
    *,
    optuna_hybrid_artifact_path: str | Path = DEFAULT_OPTUNA_HYBRID_ARTIFACT,
    integer_portfolio_artifact_path: str | Path = DEFAULT_INTEGER_PORTFOLIO_ARTIFACT,
    selected_profile_id: str = DEFAULT_SELECTED_PROFILE_ID,
) -> dict[str, Any]:
    config = load_alpha_zoo_optuna_hybrid_live_config(
        optuna_hybrid_artifact_path=optuna_hybrid_artifact_path,
        integer_portfolio_artifact_path=integer_portfolio_artifact_path,
        selected_profile_id=selected_profile_id,
    )
    max_integer_leverage = max(
        leverage
        for profile in config.source_profiles
        for leverage in profile.leverage_map.values()
    )
    risk_caps = _notional_risk_caps(config)
    real_money_blockers = [
        "paper_testnet_artifacts_only: ready_for_real, real_money_execution, and real_execution_allowed are false",
        "no_exchange_paper_fill_telemetry: realized BBO spread, fees, slippage, rejects, partial fills, and cancels are not observed yet",
        "backtest_cost_is_proxy: 10bps round-trip friction is enforced in replay and gates but is not a live measured all-in cost",
        "fail_closed_allocation: decision target_allocation is 0.0 and live sizing depends on SignalEvent.metadata.target_allocation",
    ]
    known_limitations = [
        "The selected v3.5 Optuna blend is dominated by the aggressive source profile, so independent-alpha diversification is limited.",
        "Validation MDD is near the relaxed 20% label and exceeds the strict 12% promotion cap.",
        "locked-OOS remains gate/report-only; it is not a parameter-fitting or selection surface.",
        "The live adapter uses completed 1h/2h/4h bars, so it does not model intrabar exits or exchange microstructure timing.",
        "Paper/testnet liquidity can diverge from real exchange liquidity, funding, fees, and liquidation mechanics.",
        "Frozen-artifact replay avoids online learning; stale artifacts or regime drift require a new research/paper review.",
    ]
    return {
        "artifact_kind": "alpha_zoo_optuna_hybrid_paper_testnet_live_decision",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "decision": "selected_live_mode",
        "selected_mode": "alpha_zoo_integer_leverage_optuna_hybrid",
        "strategy_name": "AlphaZooOptunaHybridLiveStrategy",
        "strategy_timeframe": "1h",
        "symbols": list(config.watch_symbols),
        "strategy_params": {
            "optuna_hybrid_artifact_path": str(config.optuna_artifact_path),
            "integer_portfolio_artifact_path": str(config.integer_artifact_path),
            "selected_profile_id": config.selected_profile_id,
            "paper_testnet_only": True,
            "allow_real_money": False,
        },
        "exchange": {
            "driver": "binance_futures",
            "name": "binance",
            "market_type": "future",
            "position_mode": "HEDGE",
            "margin_mode": "isolated",
            "leverage": int(max_integer_leverage),
            "testnet": True,
        },
        "target_allocation": 0.0,
        "target_allocation_mode": "notional_fraction",
        "sizing_mode": "notional_fraction",
        "risk_caps": risk_caps,
        "live_replay_sizing_contract": {
            "sizing_mode": "notional_fraction",
            "target_allocation_source": "SignalEvent.metadata.target_allocation",
            "target_allocation_meaning": "notional_fraction_of_account_equity",
            "target_notional_formula": "source_allocation_fraction*sum(final_profile_weight*integer_leverage)",
            "exchange_leverage_cap": int(max_integer_leverage),
            "fixed_dollar_max_order_value_applies": False,
            "absolute_cap_policy": "only explicit positive max_order_value is an emergency ceiling",
        },
        "window_seconds": 3600,
        "ingest_window_seconds": 3600,
        "decision_cadence_seconds": 3600,
        "paper_testnet_only": True,
        "ready_for_paper": True,
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "research_primary_round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "return_per_turnover_threshold_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "locked_oos_role": "gate_report_only",
        "replay_live_notional_parity": True,
        "real_money_blockers": real_money_blockers,
        "known_limitations": known_limitations,
        "paper_testnet_validation_requirements": [
            "realized BBO spread and all-in round-trip cost by symbol/timeframe",
            "order reject, timeout, cancel, and partial-fill rates",
            "replay/live notional parity from SignalEvent metadata to submitted order notional",
            "position reconciliation drift and stale-data blocks/recoveries",
            "liquidation-inclusive MDD and account-wipeout telemetry",
            "minimum 2 weeks paper/testnet observation before any real-money review",
        ],
        "operator_warning": "paper/testnet only; real-money startup remains vetoed by artifacts",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--optuna-hybrid-artifact", default=str(DEFAULT_OPTUNA_HYBRID_ARTIFACT))
    parser.add_argument("--integer-portfolio-artifact", default=str(DEFAULT_INTEGER_PORTFOLIO_ARTIFACT))
    parser.add_argument("--selected-profile-id", default=DEFAULT_SELECTED_PROFILE_ID)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print the decision payload summary without writing files.",
    )
    args = parser.parse_args(argv)

    payload = build_decision_payload(
        optuna_hybrid_artifact_path=args.optuna_hybrid_artifact,
        integer_portfolio_artifact_path=args.integer_portfolio_artifact,
        selected_profile_id=args.selected_profile_id,
    )
    if args.check_only:
        print(
            json.dumps(
                {
                    "strategy_name": payload["strategy_name"],
                    "paper_testnet_only": payload["paper_testnet_only"],
                    "ready_for_real": payload["ready_for_real"],
                    "real_money_execution": payload["real_money_execution"],
                    "real_execution_allowed": payload["real_execution_allowed"],
                    "research_primary_round_trip_cost_bps": payload[
                        "research_primary_round_trip_cost_bps"
                    ],
                    "symbols": payload["symbols"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Alpha Zoo Optuna Hybrid Paper/Testnet Live Decision",
                "",
                f"- strategy_name: `{payload['strategy_name']}`",
                f"- selected_mode: `{payload['selected_mode']}`",
                f"- selected_profile_id: `{payload['strategy_params']['selected_profile_id']}`",
                f"- symbols: `{', '.join(payload['symbols'])}`",
                f"- target_allocation_mode: `{payload['target_allocation_mode']}`",
                "- target_allocation source: `SignalEvent.metadata.target_allocation`",
                f"- risk_caps: `{json.dumps(payload['risk_caps'], sort_keys=True)}`",
                "- ready_for_real: `false`",
                "- real_money_execution: `false`",
                "- real_execution_allowed: `false`",
                "- primary round-trip cost: `10bps`",
                "",
                "This is a paper/testnet handoff artifact only; it is not a real-money approval.",
                "",
                "## Real-money blockers",
                "",
                *[f"- {item}" for item in payload["real_money_blockers"]],
                "",
                "## Known limitations",
                "",
                *[f"- {item}" for item in payload["known_limitations"]],
                "",
                "## Paper/testnet validation requirements",
                "",
                *[f"- {item}" for item in payload["paper_testnet_validation_requirements"]],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
