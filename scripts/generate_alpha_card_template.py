import argparse
import os
import re
from datetime import UTC, datetime

from lumina_quant.configuration.loader import load_runtime_config

try:
    from scripts.generate_promotion_gate_report import resolve_promotion_gate_inputs
except Exception:
    from generate_promotion_gate_report import resolve_promotion_gate_inputs


def _slugify(name):
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(name or "strategy")).strip("_")
    return token.lower() or "strategy"


def build_alpha_card_markdown(strategy_name, runtime, promotion_inputs):
    trading = runtime.trading
    risk = runtime.risk
    live = runtime.live
    exchange = live.exchange

    lines = [
        f"# Alpha Card - {strategy_name}",
        "",
        f"- Generated At (UTC): {datetime.now(UTC).isoformat()}",
        f"- Strategy: {strategy_name}",
        f"- Universe: {', '.join(trading.symbols)}",
        f"- Timeframe: {trading.timeframe}",
        f"- Market Type: {exchange.market_type}",
        f"- Position Mode: {exchange.position_mode}",
        "",
        "## 1) Thesis",
        "- Hypothesis:",
        "- Expected edge source:",
        "- Invalidation condition:",
        "",
        "## 2) Signal Contract",
        "- Inputs/features:",
        "- Entry rule:",
        "- Exit rule:",
        "- Holding horizon:",
        "",
        "## 3) Sizing & Risk",
        f"- Risk per trade: {risk.risk_per_trade}",
        f"- Max daily loss pct: {risk.max_daily_loss_pct}",
        f"- Max intraday drawdown pct: {risk.max_intraday_drawdown_pct}",
        f"- Max rolling 1h loss pct: {risk.max_rolling_loss_pct_1h}",
        f"- Max total margin pct: {risk.max_total_margin_pct}",
        f"- Max symbol exposure pct: {risk.max_symbol_exposure_pct}",
        f"- Max order value: {risk.max_order_value}",
        "",
        "## 4) Execution Assumptions",
        f"- Exchange driver/name: {exchange.driver}/{exchange.name}",
        f"- Poll interval sec: {live.poll_interval}",
        f"- Order timeout sec: {live.order_timeout}",
        f"- Reconciliation interval sec: {live.reconciliation_interval_sec}",
        "",
        "## 5) Validation Artifacts",
        "- Backtest report path:",
        "- Walk-forward report path:",
        "- Cost stress test notes:",
        "- Regime split notes:",
        "",
        "## 6) Promotion Gate Profile",
        f"- Profile source: {promotion_inputs.get('profile_source', 'n/a')}",
        f"- Window days: {promotion_inputs.get('days')}",
        f"- Max order rejects: {promotion_inputs.get('max_order_rejects')}",
        f"- Max order timeouts: {promotion_inputs.get('max_order_timeouts')}",
        f"- Max reconciliation alerts: {promotion_inputs.get('max_reconciliation_alerts')}",
        f"- Max critical errors: {promotion_inputs.get('max_critical_errors')}",
        f"- Require alpha card for gate: {promotion_inputs.get('require_alpha_card')}",
        "",
        "## 7) Operator Sign-off",
        "- Owner:",
        "- Reviewer:",
        "- Date:",
        "- Decision: HOLD / PAPER / PROMOTE",
    ]
    return "\n".join(lines)


def generate_alpha_card(
    config_path="config.yaml", strategy_name=None, output_path="", overwrite=False
):
    runtime = load_runtime_config(config_path=config_path)
    strategy = (
        str(strategy_name or runtime.optimization.strategy or "Strategy").strip() or "Strategy"
    )
    promotion_inputs = resolve_promotion_gate_inputs(
        config_path=config_path, strategy_name=strategy
    )

    markdown = build_alpha_card_markdown(strategy, runtime, promotion_inputs)

    out = (output_path or "").strip()
    if not out:
        os.makedirs("reports", exist_ok=True)
        out = os.path.join("reports", f"alpha_card_{_slugify(strategy)}.md")
    else:
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)

    if os.path.exists(out) and not overwrite:
        raise FileExistsError(f"Alpha card already exists: {out}. Use --overwrite to replace.")

    with open(out, "w", encoding="utf-8") as f:
        f.write(markdown)

    return out, markdown


def main():
    parser = argparse.ArgumentParser(description="Generate Alpha Card template markdown.")
    parser.add_argument("--config", default="config.yaml", help="Runtime config path")
    parser.add_argument(
        "--strategy", default="", help="Strategy name (defaults to optimization.strategy)"
    )
    parser.add_argument("--output", default="", help="Output markdown path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args()

    out_path, markdown = generate_alpha_card(
        config_path=args.config,
        strategy_name=(args.strategy or None),
        output_path=args.output,
        overwrite=bool(args.overwrite),
    )
    print(markdown)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
