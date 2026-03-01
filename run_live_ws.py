import argparse
import os

from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.config import LiveConfig
from lumina_quant.live.execution_live import LiveExecutionHandler
from lumina_quant.live.trader import LiveTrader
from lumina_quant.live_selection import (
    extract_selection_config,
    infer_strategy_class_name,
    load_selection_payload,
    resolve_selection_file,
)
from lumina_quant.strategies import (
    DEFAULT_STRATEGY_NAME,
    get_live_strategy_map,
    resolve_strategy_class,
)

STRATEGY_MAP = get_live_strategy_map(include_opt_in=True)
DEFAULT_WS_STRATEGY_NAME = "RsiStrategy"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LuminaQuant live trader (WebSocket).")
    parser.add_argument(
        "--enable-live-real",
        action="store_true",
        help="Explicitly allow real trading mode.",
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGY_MAP.keys()),
        default="",
        help="Strategy class override. If omitted, live selection artifact is used when available.",
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
    args = parser.parse_args()
    if args.enable_live_real:
        os.environ["LUMINA_ENABLE_LIVE_REAL"] = "true"

    from lumina_quant.live.data_ws import BinanceWebSocketDataHandler

    symbol_list = list(LiveConfig.SYMBOLS)
    strategy_params = {}
    strategy_cls = resolve_strategy_class(
        DEFAULT_WS_STRATEGY_NAME,
        default_name=DEFAULT_STRATEGY_NAME,
    )
    strategy_name = strategy_cls.__name__
    selection_path = None
    selection_cfg = None

    if not bool(args.no_selection):
        selection_path = resolve_selection_file(args.selection_file)
        if selection_path is not None:
            selection_payload = load_selection_payload(selection_path)
            selection_cfg = extract_selection_config(selection_payload)
            candidate_name = str(selection_cfg.get("candidate_name") or "").strip()
            inferred_name = infer_strategy_class_name(candidate_name)
            if inferred_name and inferred_name in STRATEGY_MAP:
                strategy_cls = STRATEGY_MAP[inferred_name]
                strategy_name = inferred_name
            selected_symbols = list(selection_cfg.get("symbols") or [])
            if selected_symbols:
                symbol_list = selected_symbols
            selected_timeframe = selection_cfg.get("strategy_timeframe")
            if selected_timeframe:
                LiveConfig.TIMEFRAME = str(selected_timeframe)
            selected_params = selection_cfg.get("params")
            if isinstance(selected_params, dict):
                strategy_params = dict(selected_params)

    manual_strategy = str(args.strategy or "").strip()
    if manual_strategy:
        strategy_cls = STRATEGY_MAP.get(manual_strategy, strategy_cls)
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

    LiveConfig.SYMBOLS = list(symbol_list)
    LiveConfig.validate()

    print(f"Mode: {'TESTNET/PAPER' if LiveConfig.IS_TESTNET else 'REAL TRADING'}")
    print(f"Exchange: {LiveConfig.EXCHANGE}")

    if selection_path is not None:
        print(f"Selection File: {selection_path}")
        if selection_cfg is not None:
            print(f"Selection Candidate: {selection_cfg.get('candidate_name')}")
    print(f"Trading Symbols: {symbol_list}")
    print(f"Strategy Timeframe: {LiveConfig.TIMEFRAME}")
    print(f"Strategy Params: {strategy_params}")

    trader = LiveTrader(
        symbol_list=symbol_list,
        data_handler_cls=BinanceWebSocketDataHandler,  # Using WS Handler
        execution_handler_cls=LiveExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=strategy_cls,
        strategy_params=strategy_params,
        strategy_name=strategy_name,
        stop_file=args.stop_file,
        external_run_id=args.run_id,
    )

    trader.run()
