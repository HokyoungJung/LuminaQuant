#!/usr/bin/env python3
"""Write the 69-asset efficiency-repair paper/testnet live decision."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from lumina_quant.strategies.alpha_zoo_optuna_hybrid_live import (
    DEFAULT_69_ASSET_EFFICIENCY_REPAIR_ARTIFACT,
    DEFAULT_INTEGER_PORTFOLIO_ARTIFACT,
    DEFAULT_SELECTED_PROFILE_ID,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_base_decision_builder():
    path = REPO_ROOT / "scripts" / "ops" / "write_alpha_zoo_optuna_hybrid_live_decision.py"
    spec = importlib.util.spec_from_file_location(
        "write_alpha_zoo_optuna_hybrid_live_decision", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base live decision writer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_decision_payload


DEFAULT_OUTPUT_DIR = (
    Path("var")
    / "reports"
    / "profit_moonshot_20260501"
    / "current_tail_20260508"
    / "alpha_v2"
    / "alpha_zoo_69_asset_efficiency_repair_optuna_20260530"
)
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "paper_testnet_live_decision_latest.json"


def _artifact_default_selected_profile_id(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = dict(payload.get("selected_optuna_hybrid_profile") or {})
    profile_id = str(selected.get("profile_id") or "").strip()
    return profile_id or DEFAULT_SELECTED_PROFILE_ID


def build_69_asset_efficiency_repair_decision_payload(
    *,
    optuna_hybrid_artifact_path: str | Path = DEFAULT_69_ASSET_EFFICIENCY_REPAIR_ARTIFACT,
    integer_portfolio_artifact_path: str | Path = DEFAULT_INTEGER_PORTFOLIO_ARTIFACT,
    selected_profile_id: str | None = None,
) -> dict[str, Any]:
    resolved_selected_profile_id = (
        str(selected_profile_id).strip()
        if selected_profile_id is not None and str(selected_profile_id).strip()
        else _artifact_default_selected_profile_id(optuna_hybrid_artifact_path)
    )
    build_decision_payload = _load_base_decision_builder()
    return build_decision_payload(
        optuna_hybrid_artifact_path=optuna_hybrid_artifact_path,
        integer_portfolio_artifact_path=integer_portfolio_artifact_path,
        selected_profile_id=resolved_selected_profile_id,
    )


def _write_markdown(output: Path, payload: dict[str, Any]) -> None:
    applicability = dict(payload.get("asset_applicability_contract") or {})
    selected_source_symbols = list(applicability.get("selected_source_symbols") or [])
    output.with_suffix(".md").write_text(
        "\n".join(
            [
                "# 69-Asset Efficiency-Repair Paper/Testnet Live Decision",
                "",
                f"- strategy_name: `{payload['strategy_name']}`",
                f"- selected_mode: `{payload['selected_mode']}`",
                f"- selected_profile_id: `{payload['strategy_params']['selected_profile_id']}`",
                f"- symbol_count: `{len(payload['symbols'])}`",
                f"- selected_source_symbol_count: `{len(selected_source_symbols)}`",
                f"- selected_source_symbols: `{', '.join(selected_source_symbols)}`",
                f"- live_final_gross_notional: `{payload['live_final_weight_gross_notional_fraction']:.6f}x`",
                f"- historical_train_validation_gross_notional: `{payload['historical_train_validation_gross_notional_fraction']:.6f}x`",
                f"- target_allocation_mode: `{payload['target_allocation_mode']}`",
                "- target_allocation source: `SignalEvent.metadata.target_allocation`",
                f"- risk_caps: `{json.dumps(payload['risk_caps'], sort_keys=True)}`",
                "- limit order contract: `LMT one_tick_worse by default; market optional only by explicit opt-in`",
                "- unfilled-order policy: `cancel, reconcile, do not chase, no market fallback`",
                "- slippage guard: `missing-BBO/guard breach skip or cancel; no high-slippage fallback`",
                "- ready_for_real: `false`",
                "- real_money_execution: `false`",
                "- real_execution_allowed: `false`",
                "- primary round-trip cost: `10bps`",
                "",
                "This is a paper/testnet handoff artifact only; it is not a real-money approval.",
                "",
                "## No-fill / slippage policy",
                "",
                f"```json\n{json.dumps(payload['unfilled_order_policy'], indent=2, sort_keys=True)}\n```",
                "",
                f"```json\n{json.dumps(payload['slippage_guard_policy'], indent=2, sort_keys=True)}\n```",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--optuna-hybrid-artifact", default=str(DEFAULT_69_ASSET_EFFICIENCY_REPAIR_ARTIFACT)
    )
    parser.add_argument(
        "--integer-portfolio-artifact", default=str(DEFAULT_INTEGER_PORTFOLIO_ARTIFACT)
    )
    parser.add_argument(
        "--selected-profile-id",
        default="",
        help="Override selected profile id; default uses artifact selected_optuna_hybrid_profile.",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    payload = build_69_asset_efficiency_repair_decision_payload(
        optuna_hybrid_artifact_path=args.optuna_hybrid_artifact,
        integer_portfolio_artifact_path=args.integer_portfolio_artifact,
        selected_profile_id=args.selected_profile_id,
    )
    if args.check_only:
        print(
            json.dumps(
                {
                    "selected_mode": payload["selected_mode"],
                    "strategy_name": payload["strategy_name"],
                    "paper_testnet_only": payload["paper_testnet_only"],
                    "ready_for_real": payload["ready_for_real"],
                    "real_money_execution": payload["real_money_execution"],
                    "real_execution_allowed": payload["real_execution_allowed"],
                    "symbol_count": len(payload["symbols"]),
                    "unfilled_market_fallback_allowed": payload["unfilled_order_policy"][
                        "market_fallback_allowed"
                    ],
                    "slippage_market_fallback_allowed": payload["slippage_guard_policy"][
                        "market_fallback_allowed"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
