from __future__ import annotations

import argparse
import os

from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.configuration import get_default_runtime_config, validate_runtime_config
from lumina_quant.core.market_window_contract import MarketWindowContractError
from lumina_quant.live_selection import (
    extract_live_decision_config,
    extract_selection_config,
    infer_strategy_class_name,
    load_live_decision_payload,
    load_selection_payload,
    resolve_live_decision_file,
    resolve_selection_file,
    supports_live_portfolio_mode,
)
from lumina_quant.system_assembly import build_live_runtime_contract
from lumina_quant.strategies.artifact_portfolio_mode import (
    ArtifactPortfolioModeStrategy,
    resolve_portfolio_mode_definition,
)

DEFAULT_LIVE_STRATEGY_NAME = "MovingAverageCrossStrategy"
DEFAULT_WS_STRATEGY_NAME = "RsiStrategy"

# Module-level compatibility attribute — not used by main() which reads from
# get_default_runtime_config().  Present so tests can monkeypatch it without
# raising AttributeError (monkeypatch.setattr raising=True default).
LiveConfig = None
STRATEGY_MAP = None
resolve_strategy_class = None


def _strategy_helpers():
    global STRATEGY_MAP, resolve_strategy_class
    from lumina_quant.strategies import registry

    default_name = registry.DEFAULT_STRATEGY_NAME

    def _get_live_strategy_map(include_opt_in=True):
        if hasattr(registry, "get_live_strategy_map"):
            return registry.get_live_strategy_map(include_opt_in=include_opt_in)
        return registry.get_strategy_map()

    def _resolve_strategy_class(
        name: str,
        default_name: str | None = None,
        default_name_override: str | None = None,
    ):
        fallback_name = default_name_override if default_name_override is not None else default_name
        return registry.resolve_strategy_class(
            name,
            default_name=fallback_name or default_name,
        )

    if STRATEGY_MAP is None:
        STRATEGY_MAP = _get_live_strategy_map(include_opt_in=True)
    if resolve_strategy_class is None:
        resolve_strategy_class = _resolve_strategy_class
    return default_name, _get_live_strategy_map, resolve_strategy_class


def _resolve_transport(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {"", "poll", "ws"}:
        return token or "poll"
    raise ValueError(f"Unsupported --transport: {value}")


def _resolve_effective_transport(*, transport: str, market_data_source: str) -> str:
    selected_transport = str(transport or "poll").strip().lower() or "poll"
    selected_source = str(market_data_source or "committed").strip().lower() or "committed"
    if selected_source == "committed" and selected_transport == "ws":
        return "poll"
    return selected_transport


def _resolve_live_banner(live_cfg) -> dict[str, object]:
    """Resolve the operator-facing live banner (stage / endpoint / mode / real-money).

    Stage drives the exchange endpoint: ``canary``/``full`` route to the PRODUCTION
    endpoint (validate.py prod-routing gate, which also requires ``mode='real'``).
    An explicit ``live.testnet=False`` with ``mode='real'`` is likewise a production
    endpoint. The banner must reflect the resolved endpoint, never just the mode
    string — a banner that says PAPER while orders route to prod is a dangerous lie.
    """
    stage = str(getattr(live_cfg, "go_live_stage", "testnet")).strip().lower()
    mode = str(getattr(live_cfg, "mode", "paper")).strip().lower()
    endpoint_is_prod = stage in {"canary", "full"} or (
        getattr(live_cfg, "testnet", None) is False and mode == "real"
    )
    return {
        "stage": stage,
        "mode": "REAL" if mode == "real" else "PAPER",
        "endpoint": "PRODUCTION (live exchange)" if endpoint_is_prod else "TESTNET",
        "endpoint_is_prod": endpoint_is_prod,
    }


def _shutdown_on_fatal(trader, exc: Exception) -> None:
    print(f"\nCritical live-data contract breach: {exc}")
    ordered_shutdown = getattr(trader, "_ordered_shutdown", None)
    if callable(ordered_shutdown):
        ordered_shutdown()
    close_audit = getattr(trader, "_close_audit_store", None)
    if callable(close_audit):
        close_audit(status="FAILED")
    raise SystemExit(2) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LuminaQuant live trader.")
    parser.add_argument(
        "--transport",
        choices=["poll", "ws"],
        default=str(os.getenv("LQ_LIVE_TRANSPORT", "poll") or "poll"),
        help="Live market-data transport (poll or ws).",
    )
    parser.add_argument(
        "--enable-live-real",
        action="store_true",
        help="Explicitly allow real trading mode.",
    )
    parser.add_argument(
        "--strategy",
        default="",
        help="Strategy class override. If omitted, live selection artifact is used when available.",
    )
    parser.add_argument(
        "--decision-file",
        default="",
        help=(
            "Optional live decision JSON path. If omitted, the latest followup-status "
            "live-readiness decision is used when available."
        ),
    )
    parser.add_argument(
        "--selection-file",
        default="",
        help=(
            "Optional live selection JSON path. If omitted, newest file under "
            "best_optimized_parameters/live/ is used."
        ),
    )
    parser.add_argument(
        "--no-selection",
        action="store_true",
        help="Disable loading live selection artifact and run with config/manual strategy only.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional external run_id for audit trail correlation.",
    )
    parser.add_argument(
        "--stop-file",
        default="",
        help="Optional stop-file path for graceful shutdown signal.",
    )
    args = parser.parse_args(argv)

    default_strategy_registry_name, get_live_strategy_map, resolve_strategy_class = (
        _strategy_helpers()
    )
    strategy_map = (
        STRATEGY_MAP
        if isinstance(STRATEGY_MAP, dict)
        else get_live_strategy_map(include_opt_in=True)
    )

    transport = _resolve_transport(args.transport)
    if args.enable_live_real:
        os.environ["LUMINA_ENABLE_LIVE_REAL"] = "true"

    # 1. Check Configuration
    print("=== LuminaQuant Live Trader ===")

    # 2. Setup — load typed RuntimeConfig once; mutate fields in-place from decision artifacts
    rt = get_default_runtime_config()
    symbol_list = list(rt.trading.symbols)
    resolved_timeframe = str(rt.trading.timeframe)
    strategy_params = {}
    default_strategy_name = (
        DEFAULT_WS_STRATEGY_NAME if transport == "ws" else DEFAULT_LIVE_STRATEGY_NAME
    )
    strategy_cls = resolve_strategy_class(
        default_strategy_name,
        default_name=default_strategy_registry_name,
    )
    strategy_name = strategy_cls.__name__
    manual_strategy = str(args.strategy or "").strip()
    decision_path = None
    decision_cfg = None
    selection_path = None
    selection_cfg = None
    LiveDataFatalError = RuntimeError
    trader = None

    try:
        if not bool(args.no_selection):
            decision_path = resolve_live_decision_file(args.decision_file)
        if decision_path is not None and not manual_strategy:
            decision_payload = load_live_decision_payload(decision_path)
            decision_cfg = extract_live_decision_config(decision_payload)
            target_kind = str(decision_cfg.get("target_kind") or "")
            if target_kind == "strategy_class":
                inferred_name = str(decision_cfg.get("strategy_name") or "").strip()
                if not inferred_name or inferred_name not in strategy_map:
                    raise ValueError(
                        "Live decision points to unsupported strategy class "
                        f"'{inferred_name or 'unknown'}'. Pass an explicit --strategy override "
                        "or register the strategy for live execution."
                    )
                strategy_cls = strategy_map[inferred_name]
                strategy_name = inferred_name
                decision_symbols = list(decision_cfg.get("symbols") or [])
                if decision_symbols:
                    symbol_list = [str(item) for item in decision_symbols]
                decision_timeframe = decision_cfg.get("strategy_timeframe")
                if decision_timeframe:
                    resolved_timeframe = str(decision_timeframe)
                decision_params = decision_cfg.get("strategy_params")
                if isinstance(decision_params, dict):
                    strategy_params = dict(decision_params)
            elif target_kind == "portfolio_mode":
                reference = (
                    str(decision_cfg.get("reference") or "").strip() or "unknown_portfolio_mode"
                )
                if not supports_live_portfolio_mode(reference):
                    raise ValueError(
                        "Live decision points to unsupported portfolio mode "
                        f"'{reference}'. Pass an explicit --strategy override or extend the portfolio-mode runtime."
                    )
                mode_definition = resolve_portfolio_mode_definition(reference)
                strategy_cls = ArtifactPortfolioModeStrategy
                strategy_name = f"ArtifactPortfolioModeStrategy[{reference}]"
                strategy_params = {"portfolio_mode": reference}
                symbol_list = list(mode_definition.symbols)

            exchange_override = decision_cfg.get("exchange")
            if isinstance(exchange_override, dict) and exchange_override:
                if "name" in exchange_override:
                    rt.live.exchange.name = str(exchange_override["name"])
                if "market_type" in exchange_override:
                    rt.live.exchange.market_type = str(exchange_override["market_type"])
                if "position_mode" in exchange_override:
                    rt.live.exchange.position_mode = str(exchange_override["position_mode"])
                if "margin_mode" in exchange_override:
                    rt.live.exchange.margin_mode = str(exchange_override["margin_mode"])
                if "leverage" in exchange_override:
                    rt.live.exchange.leverage = int(exchange_override["leverage"])
            elif decision_cfg.get("leverage") is not None:
                rt.live.exchange.leverage = int(decision_cfg["leverage"])

            if decision_cfg.get("target_allocation") is not None:
                rt.trading.target_allocation = float(decision_cfg["target_allocation"])
            if decision_cfg.get("target_allocation_mode") is not None:
                rt.trading.target_allocation_mode = str(decision_cfg["target_allocation_mode"])
            for cfg_key in (
                "max_order_value",
                "max_order_notional_pct",
                "max_symbol_exposure_pct",
                "max_total_margin_pct",
                "max_total_notional_pct",
            ):
                if decision_cfg.get(cfg_key) is not None:
                    setattr(rt.risk, cfg_key, float(decision_cfg[cfg_key]))
            if decision_cfg.get("window_seconds") is not None:
                window_seconds = int(decision_cfg["window_seconds"])
                rt.live.window_seconds = window_seconds
                if decision_cfg.get("ingest_window_seconds") is None:
                    rt.live.ingest_window_seconds = window_seconds
            if decision_cfg.get("ingest_window_seconds") is not None:
                rt.live.ingest_window_seconds = int(decision_cfg["ingest_window_seconds"])
            if decision_cfg.get("decision_cadence_seconds") is not None:
                rt.live.decision_cadence_seconds = int(decision_cfg["decision_cadence_seconds"])

        if not bool(args.no_selection) and (
            decision_cfg is None
            or str(decision_cfg.get("target_kind") or "") == "incumbent_fallback"
        ):
            selection_path = resolve_selection_file(args.selection_file)
            if selection_path is not None:
                selection_payload = load_selection_payload(selection_path)
                selection_cfg = extract_selection_config(selection_payload)
                candidate_name = str(selection_cfg.get("candidate_name") or "").strip()
                inferred_name = infer_strategy_class_name(candidate_name)
                if inferred_name and inferred_name in strategy_map:
                    strategy_cls = strategy_map[inferred_name]
                    strategy_name = inferred_name
                selected_symbols = list(selection_cfg.get("symbols") or [])
                if selected_symbols:
                    symbol_list = selected_symbols
                selected_timeframe = selection_cfg.get("strategy_timeframe")
                if selected_timeframe:
                    resolved_timeframe = str(selected_timeframe)
                selected_params = selection_cfg.get("params")
                if isinstance(selected_params, dict):
                    strategy_params = dict(selected_params)

        if manual_strategy:
            if manual_strategy not in strategy_map:
                raise ValueError(f"Unknown strategy override: {manual_strategy}")
            strategy_cls = strategy_map[manual_strategy]
            strategy_name = manual_strategy
            if selection_cfg is not None:
                inferred_name = infer_strategy_class_name(
                    str(selection_cfg.get("candidate_name") or "")
                )
                if inferred_name != manual_strategy:
                    strategy_params = {}
                    print(
                        "[WARN] --strategy overrides selection strategy. "
                        "Selection params were ignored to avoid mismatch."
                    )
    except Exception as exc:
        print(f"\nCritical Error: {exc}")
        return 1

    rt.trading.symbols = list(symbol_list)
    rt.trading.timeframe = str(resolved_timeframe)
    validate_runtime_config(rt, for_live=True)

    # Operator-facing banner must reflect the ACTUAL resolved endpoint, not just
    # the mode string — a banner that says PAPER while orders route to prod is a
    # dangerous lie on a real-money tool.
    banner = _resolve_live_banner(rt.live)
    print(f"Go-Live Stage: {banner['stage']}")
    print(f"Endpoint: {banner['endpoint']}")
    print(f"Mode: {banner['mode']}")
    if banner["endpoint_is_prod"]:
        print("⚠  REAL MONEY — orders route to the PRODUCTION exchange endpoint")
    print(f"Exchange: {rt.live.exchange.name} ({rt.live.exchange.market_type})")
    market_data_source = str(rt.live.market_data_source)
    order_state_source = str(rt.live.order_state_source)
    effective_transport = _resolve_effective_transport(
        transport=transport,
        market_data_source=market_data_source,
    )
    print(f"Market Data Source: {market_data_source}")
    print(f"Order State Source: {order_state_source}")
    print(f"Requested Transport: {transport}")
    print(f"Effective Transport: {effective_transport}")
    if effective_transport != transport:
        print(
            "[WARN] --transport=ws is ignored when live.market_data_source=committed "
            "(committed reader path remains active)."
        )

    if selection_path is not None:
        print(f"Selection File: {selection_path}")
        if selection_cfg is not None:
            print(f"Selection Candidate: {selection_cfg.get('candidate_name')}")
    if decision_path is not None:
        print(f"Decision File: {decision_path}")
        if decision_cfg is not None:
            print(
                f"Decision Target: {decision_cfg.get('reference') or decision_cfg.get('decision')}"
            )

    print(f"Trading Symbols: {symbol_list}")
    print(f"Strategy Timeframe: {rt.trading.timeframe}")
    print(
        "Materialized Staleness Gate: "
        f"threshold={rt.live.materialized_staleness_threshold_seconds}s, "
        f"alert_cooldown={rt.live.materialized_staleness_alert_cooldown_seconds}s"
    )
    print(f"Strategy Params: {strategy_params}")

    try:
        live_runtime = build_live_runtime_contract(transport=effective_transport)
        LiveTrader = live_runtime.engine_cls
        LiveDataFatalError = live_runtime.fatal_error_cls
        data_handler_cls = live_runtime.data_handler_cls
        execution_handler_cls = live_runtime.execution_handler_cls
        portfolio_cls = live_runtime.portfolio_cls

        trader = LiveTrader(
            symbol_list=symbol_list,
            data_handler_cls=data_handler_cls,
            execution_handler_cls=execution_handler_cls,
            portfolio_cls=portfolio_cls,
            strategy_cls=strategy_cls,
            strategy_params=strategy_params,
            strategy_name=strategy_name,
            stop_file=args.stop_file,
            external_run_id=args.run_id,
            runtime_config=rt,
        )

        print("Starting engine... Press Ctrl+C to stop.")
        trader.run()
        consume = getattr(trader.data_handler, "consume_fatal_error", None)
        if callable(consume):
            fatal_exc = consume()
            if fatal_exc is not None:
                _shutdown_on_fatal(trader, fatal_exc)

    except KeyboardInterrupt:
        print("\nStopping trader...")
        return 130
    except (RawFirstDataMissingError, MarketWindowContractError, LiveDataFatalError) as exc:
        if trader is not None:
            _shutdown_on_fatal(trader, exc)
        return 2
    except Exception as exc:
        print(f"\nCritical Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
