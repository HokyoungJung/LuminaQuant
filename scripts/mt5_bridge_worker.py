"""MetaTrader5 bridge worker.

Run this script with a Windows Python that has MetaTrader5 installed.
It executes one action per invocation and returns JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


def _emit(ok: bool, *, result: Any = None, error: str = "") -> None:
    payload = {
        "ok": bool(ok),
        "result": result,
        "error": str(error or ""),
    }
    print(json.dumps(payload, ensure_ascii=True))


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _initialize(payload: dict[str, Any]) -> None:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not installed in bridge environment")
    if not mt5.initialize():
        raise RuntimeError(f"initialize() failed: {mt5.last_error()}")

    login = payload.get("login")
    password = payload.get("password")
    server = payload.get("server")
    if login and password and server:
        ok = mt5.login(_to_int(login), password=str(password), server=str(server))
        if not ok:
            raise RuntimeError(f"login() failed: {mt5.last_error()}")


def _action_connect(payload: dict[str, Any]) -> dict[str, Any]:
    _ = payload
    return {
        "connected": True,
        "version": str(getattr(mt5, "__version__", "unknown")),
    }


def _action_get_balance(payload: dict[str, Any]) -> float:
    _ = payload
    account_info = mt5.account_info()
    if account_info is None:
        return 0.0
    return float(getattr(account_info, "balance", 0.0) or 0.0)


def _action_get_all_positions(payload: dict[str, Any]) -> dict[str, float]:
    _ = payload
    positions = mt5.positions_get()
    if positions is None:
        return {}
    out: dict[str, float] = {}
    for pos in positions:
        symbol = str(getattr(pos, "symbol", "") or "")
        volume = _to_float(getattr(pos, "volume", 0.0), 0.0)
        if not symbol:
            continue
        out[symbol] = float(out.get(symbol, 0.0) + volume)
    return out


def _action_fetch_ohlcv(payload: dict[str, Any]) -> list[list[float]]:
    symbol = str(payload.get("symbol") or "")
    timeframe = str(payload.get("timeframe") or "1m")
    limit = max(1, _to_int(payload.get("limit"), 100))

    tf_map = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
    mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, limit)
    if rates is None:
        return []

    out: list[list[float]] = []
    for rate in rates:
        out.append(
            [
                float(_to_int(rate["time"]) * 1000),
                _to_float(rate["open"]),
                _to_float(rate["high"]),
                _to_float(rate["low"]),
                _to_float(rate["close"]),
                _to_float(rate["tick_volume"]),
            ]
        )
    return out


def _action_execute_order(payload: dict[str, Any]) -> dict[str, Any]:
    symbol = str(payload.get("symbol") or "")
    order_type = str(payload.get("type") or "market").strip().lower()
    side = str(payload.get("side") or "buy").strip().lower()
    quantity = _to_float(payload.get("quantity"), 0.0)
    price_raw = payload.get("price")
    price = None if price_raw is None else _to_float(price_raw)
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    action = mt5.TRADE_ACTION_DEAL
    mt5_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL

    if order_type == "limit":
        action = mt5.TRADE_ACTION_PENDING
        mt5_type = mt5.ORDER_TYPE_BUY_LIMIT if side == "buy" else mt5.ORDER_TYPE_SELL_LIMIT

    if order_type == "market":
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Symbol {symbol} not found")
        price = (
            _to_float(getattr(tick, "ask", 0.0))
            if side == "buy"
            else _to_float(getattr(tick, "bid", 0.0))
        )

    magic = _to_int(params.get("magic", payload.get("mt5_magic", 234000)), 234000)
    deviation = _to_int(params.get("deviation", payload.get("mt5_deviation", 20)), 20)
    comment = str(params.get("comment", payload.get("mt5_comment", "LuminaQuant")))
    filler_type = _to_int(params.get("type_filling", mt5.ORDER_FILLING_IOC), mt5.ORDER_FILLING_IOC)
    sl = _to_float(params.get("sl", 0.0), 0.0)
    tp = _to_float(params.get("tp", 0.0), 0.0)

    request = {
        "action": action,
        "symbol": symbol,
        "volume": quantity,
        "type": mt5_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filler_type,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"Order failed: {result.comment}")
    return {
        "id": str(result.order),
        "status": "closed" if order_type == "market" else "open",
        "filled": _to_float(result.volume),
        "average": _to_float(result.price),
        "price": _to_float(result.price),
        "amount": _to_float(result.volume),
    }


def _action_fetch_order(payload: dict[str, Any]) -> dict[str, Any]:
    order_id = str(payload.get("order_id") or "")
    try:
        ticket = int(order_id)
    except Exception:
        return {}
    orders = mt5.orders_get(ticket=ticket)
    if not orders:
        return {}
    order = orders[0]
    return {
        "id": str(order.ticket),
        "status": "open",
        "filled": _to_float(order.volume_initial) - _to_float(order.volume_current),
        "average": _to_float(order.price_open),
        "price": _to_float(order.price_open),
        "amount": _to_float(order.volume_initial),
        "symbol": str(order.symbol),
    }


def _action_fetch_open_orders(payload: dict[str, Any]) -> list[dict[str, Any]]:
    symbol = payload.get("symbol")
    if symbol:
        orders = mt5.orders_get(symbol=str(symbol))
    else:
        orders = mt5.orders_get()
    if orders is None:
        return []

    out: list[dict[str, Any]] = []
    for order in orders:
        side = (
            "buy"
            if order.type
            in [
                mt5.ORDER_TYPE_BUY,
                mt5.ORDER_TYPE_BUY_LIMIT,
                mt5.ORDER_TYPE_BUY_STOP,
            ]
            else "sell"
        )
        out.append(
            {
                "id": str(order.ticket),
                "symbol": str(order.symbol),
                "type": "buy" if order.type == mt5.ORDER_TYPE_BUY else "sell",
                "side": side,
                "price": _to_float(order.price_open),
                "amount": _to_float(order.volume_initial),
                "filled": _to_float(order.volume_current),
                "filled_qty": _to_float(order.volume_initial) - _to_float(order.volume_current),
                "status": "open",
            }
        )
    return out


def _action_cancel_order(payload: dict[str, Any]) -> bool:
    order_id = str(payload.get("order_id") or "")
    try:
        ticket = int(order_id)
    except Exception:
        return False
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": ticket,
        "comment": "LuminaQuant Cancel",
    }
    result = mt5.order_send(request)
    return bool(result.retcode == mt5.TRADE_RETCODE_DONE)


def _action_disconnect(payload: dict[str, Any]) -> bool:
    _ = payload
    return True


ACTIONS: dict[str, Any] = {
    "connect": _action_connect,
    "disconnect": _action_disconnect,
    "get_balance": _action_get_balance,
    "get_all_positions": _action_get_all_positions,
    "fetch_ohlcv": _action_fetch_ohlcv,
    "execute_order": _action_execute_order,
    "fetch_order": _action_fetch_order,
    "fetch_open_orders": _action_fetch_open_orders,
    "cancel_order": _action_cancel_order,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="MT5 bridge worker")
    parser.add_argument("--action", required=True)
    parser.add_argument("--payload", default="{}")
    args = parser.parse_args()

    try:
        payload_raw = json.loads(str(args.payload))
    except Exception:
        _emit(False, error="Invalid JSON payload")
        return
    payload = payload_raw if isinstance(payload_raw, dict) else {}
    action_name = str(args.action).strip()
    handler = ACTIONS.get(action_name)
    if handler is None:
        _emit(False, error=f"Unsupported action: {action_name}")
        return

    try:
        _initialize(payload)
        result = handler(payload)
        _emit(True, result=result)
    except Exception as exc:
        _emit(False, error=str(exc))
    finally:
        if mt5 is not None:
            try:
                mt5.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
