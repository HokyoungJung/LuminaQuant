"""Promotion-gate input resolver and lightweight report helper."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from lumina_quant.configuration.loader import load_runtime_config


def resolve_promotion_gate_inputs(
    *,
    config_path: str = "config.yaml",
    strategy_name: str | None = None,
) -> dict[str, Any]:
    """Return merged promotion-gate inputs for a strategy."""
    runtime = load_runtime_config(config_path=config_path)
    strategy = str(strategy_name or runtime.optimization.strategy or "").strip()
    gate = runtime.promotion_gate
    profile = dict(gate.strategy_profiles.get(strategy, {})) if strategy else {}

    def _pick(name: str, default: Any) -> Any:
        return profile.get(name, default)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy": strategy or runtime.optimization.strategy,
        "profile_source": ("strategy_profiles" if profile else "defaults"),
        "days": int(_pick("days", gate.days)),
        "max_order_rejects": int(_pick("max_order_rejects", gate.max_order_rejects)),
        "max_order_timeouts": int(_pick("max_order_timeouts", gate.max_order_timeouts)),
        "max_reconciliation_alerts": int(
            _pick("max_reconciliation_alerts", gate.max_reconciliation_alerts)
        ),
        "max_critical_errors": int(_pick("max_critical_errors", gate.max_critical_errors)),
        "require_alpha_card": bool(_pick("require_alpha_card", gate.require_alpha_card)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Show resolved promotion-gate inputs.")
    parser.add_argument("--config", default="config.yaml", help="Runtime config path")
    parser.add_argument("--strategy", default="", help="Strategy name override")
    args = parser.parse_args()
    payload = resolve_promotion_gate_inputs(
        config_path=str(args.config),
        strategy_name=(str(args.strategy).strip() or None),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
