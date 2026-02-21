from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from lumina_quant.interfaces import ExchangeInterface


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    token = str(value).strip().lower()
    if not token:
        return bool(default)
    return token not in {"0", "false", "no", "off"}


def _is_wsl() -> bool:
    if os.name != "posix":
        return False
    try:
        with open("/proc/sys/kernel/osrelease", encoding="utf-8") as f:
            text = f.read().lower()
    except Exception:
        return False
    return "microsoft" in text or "wsl" in text


class MT5Exchange(ExchangeInterface):
    """Implementation of ExchangeInterface using MetaTrader5.

    On Windows, this uses in-process MetaTrader5 Python package.
    On WSL/Linux, set `LQ_MT5_BRIDGE_PYTHON` to a Windows Python executable path
    to use bridge mode through `scripts/mt5_bridge_worker.py`.
    """

    def __init__(self, config):
        self.config = config
        self.connected = False
        root = Path(__file__).resolve().parents[2]
        default_bridge_script = str(root / "scripts" / "mt5_bridge_worker.py")
        self._bridge_python = str(
            getattr(config, "MT5_BRIDGE_PYTHON", "") or os.getenv("LQ_MT5_BRIDGE_PYTHON", "")
        ).strip()
        self._bridge_script = str(
            getattr(config, "MT5_BRIDGE_SCRIPT", "")
            or os.getenv("LQ_MT5_BRIDGE_SCRIPT", "")
            or default_bridge_script
        ).strip()
        self._bridge_use_wslpath = _as_bool(
            getattr(config, "MT5_BRIDGE_USE_WSLPATH", None)
            if hasattr(config, "MT5_BRIDGE_USE_WSLPATH")
            else os.getenv("LQ_MT5_BRIDGE_USE_WSLPATH", "1"),
            True,
        )
        self.connect()

    def _bridge_enabled(self) -> bool:
        return bool(self._bridge_python)

    def _bridge_mode(self) -> bool:
        return mt5 is None and self._bridge_enabled()

    def _bridge_auth_payload(self) -> dict[str, Any]:
        return {
            "login": getattr(self.config, "MT5_LOGIN", None),
            "password": getattr(self.config, "MT5_PASSWORD", None),
            "server": getattr(self.config, "MT5_SERVER", None),
            "mt5_magic": getattr(self.config, "MT5_MAGIC", 234000),
            "mt5_deviation": getattr(self.config, "MT5_DEVIATION", 20),
            "mt5_comment": getattr(self.config, "MT5_COMMENT", "LuminaQuant"),
        }

    def _bridge_script_for_target(self) -> str:
        script_path = Path(self._bridge_script)
        if not script_path.is_absolute():
            script_path = Path.cwd() / script_path
        script_local = str(script_path)

        python_token = self._bridge_python.lower()
        if (
            self._bridge_use_wslpath
            and _is_wsl()
            and (python_token.endswith(".exe") or "\\\\" in python_token)
        ):
            proc = subprocess.run(
                ["wslpath", "-w", script_local],
                capture_output=True,
                text=True,
                check=False,
            )
            converted = (proc.stdout or "").strip()
            if proc.returncode == 0 and converted:
                return converted
        return script_local

    def _bridge_call(self, action: str, payload: dict[str, Any] | None = None) -> Any:
        if not self._bridge_enabled():
            raise RuntimeError("MT5 bridge is not configured.")
        merged_payload = dict(payload or {})
        merged_payload.update(self._bridge_auth_payload())

        cmd = [
            self._bridge_python,
            self._bridge_script_for_target(),
            "--action",
            str(action),
            "--payload",
            json.dumps(merged_payload, ensure_ascii=True),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"MT5 bridge command failed ({action}): {detail}")

        stdout_text = (proc.stdout or "").strip()
        if not stdout_text:
            raise RuntimeError(f"MT5 bridge command returned empty response ({action}).")
        response_line = stdout_text.splitlines()[-1]
        try:
            response = json.loads(response_line)
        except Exception as exc:
            raise RuntimeError(f"Invalid MT5 bridge response ({action}): {stdout_text}") from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected MT5 bridge payload type ({action}).")
        if not bool(response.get("ok", False)):
            error_text = str(response.get("error") or f"MT5 bridge call failed: {action}")
            raise RuntimeError(error_text)
        return response.get("result")

    def connect(self):
        if mt5 is None:
            if not self._bridge_enabled():
                print(
                    "MetaTrader5 package not installed. Set LQ_MT5_BRIDGE_PYTHON for WSL/Linux bridge mode."
                )
                return
            try:
                self._bridge_call("connect")
                self.connected = True
                print("Connected to MT5 via bridge worker.")
            except Exception as exc:
                print(f"MT5 bridge connection failed: {exc}")
                self.connected = False
            return

        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            self.connected = False
            return

        print("MetaTrader5 package version:", mt5.__version__)
        self.connected = True

        login = getattr(self.config, "MT5_LOGIN", None)
        password = getattr(self.config, "MT5_PASSWORD", None)
        server = getattr(self.config, "MT5_SERVER", None)

        if login and password and server:
            authorized = mt5.login(login, password=password, server=server)
            if authorized:
                print(f"Connected to account #{login}")
            else:
                print(f"failed to connect at account #{login}, error code: {mt5.last_error()}")

    def get_balance(self, currency: str = "USDT") -> float:
        if not self.connected:
            return 0.0
        if self._bridge_mode():
            try:
                result = self._bridge_call("get_balance", {"currency": currency})
                return float(result or 0.0)
            except Exception:
                return 0.0
        account_info = mt5.account_info()
        if account_info is None:
            return 0.0
        return float(account_info.balance)

    def get_all_positions(self) -> dict[str, float]:
        if not self.connected:
            return {}
        if self._bridge_mode():
            try:
                result = self._bridge_call("get_all_positions")
            except Exception:
                return {}
            if not isinstance(result, dict):
                return {}
            out: dict[str, float] = {}
            for key, value in result.items():
                try:
                    out[str(key)] = float(value)
                except Exception:
                    continue
            return out

        positions = mt5.positions_get()
        if positions is None:
            return {}
        result: dict[str, float] = {}
        for pos in positions:
            result[str(pos.symbol)] = float(pos.volume)
        return result

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> list[tuple]:
        if not self.connected:
            return []

        if self._bridge_mode():
            try:
                payload = {
                    "symbol": str(symbol),
                    "timeframe": str(timeframe),
                    "limit": int(limit),
                }
                result = self._bridge_call("fetch_ohlcv", payload)
            except Exception:
                return []
            if not isinstance(result, list):
                return []
            out: list[tuple] = []
            for row in result:
                if isinstance(row, (list, tuple)) and len(row) >= 6:
                    out.append(tuple(row[:6]))
            return out

        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }

        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, limit)

        if rates is None:
            print(f"Failed to get rates for {symbol}")
            return []

        data = []
        for rate in rates:
            timestamp = int(rate["time"]) * 1000
            data.append(
                (
                    timestamp,
                    float(rate["open"]),
                    float(rate["high"]),
                    float(rate["low"]),
                    float(rate["close"]),
                    float(rate["tick_volume"]),
                )
            )
        return data

    def execute_order(
        self,
        symbol: str,
        type: str,
        side: str,
        quantity: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> dict:
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
        params = params or {}

        if self._bridge_mode():
            payload = {
                "symbol": str(symbol),
                "type": str(type),
                "side": str(side),
                "quantity": float(quantity),
                "price": None if price is None else float(price),
                "params": dict(params),
            }
            result = self._bridge_call("execute_order", payload)
            if isinstance(result, dict):
                return result
            raise RuntimeError("MT5 bridge execute_order returned invalid payload")

        action = mt5.TRADE_ACTION_DEAL
        mt5_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL

        if type == "limit":
            action = mt5.TRADE_ACTION_PENDING
            if side == "buy":
                mt5_type = mt5.ORDER_TYPE_BUY_LIMIT
            else:
                mt5_type = mt5.ORDER_TYPE_SELL_LIMIT

        if type == "market":
            symbol_info = mt5.symbol_info_tick(symbol)
            if symbol_info is None:
                raise RuntimeError(f"Symbol {symbol} not found")
            if side == "buy":
                price = symbol_info.ask
            else:
                price = symbol_info.bid

        magic = params.get("magic", getattr(self.config, "MT5_MAGIC", 234000))
        deviation = params.get("deviation", getattr(self.config, "MT5_DEVIATION", 20))
        comment = params.get("comment", getattr(self.config, "MT5_COMMENT", "LuminaQuant"))
        filler_type = params.get("type_filling", mt5.ORDER_FILLING_IOC)
        sl = params.get("sl", 0.0)
        tp = params.get("tp", 0.0)

        request = {
            "action": action,
            "symbol": symbol,
            "volume": quantity,
            "type": mt5_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filler_type,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order send failed, retcode={result.retcode}")
            raise RuntimeError(f"Order failed: {result.comment}")

        return {
            "id": str(result.order),
            "status": "closed" if type == "market" else "open",
            "filled": result.volume,
            "average": result.price,
            "price": result.price,
            "amount": result.volume,
        }

    def load_markets(self) -> dict:
        return {}

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        _ = (symbol, leverage)
        return True

    def set_margin_mode(self, symbol: str, margin_mode: str) -> bool:
        _ = (symbol, margin_mode)
        return True

    def fetch_positions(self, symbol: str | None = None) -> list[dict]:
        return [
            {"symbol": sym, "contracts": qty}
            for sym, qty in self.get_all_positions().items()
            if symbol is None or sym == symbol
        ]

    def fetch_order(self, order_id: str, symbol: str | None = None) -> dict:
        _ = symbol
        if not self.connected:
            return {}
        if self._bridge_mode():
            try:
                result = self._bridge_call(
                    "fetch_order", {"order_id": str(order_id), "symbol": symbol}
                )
            except Exception:
                return {}
            return result if isinstance(result, dict) else {}

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
            "filled": order.volume_initial - order.volume_current,
            "average": order.price_open,
            "price": order.price_open,
            "amount": order.volume_initial,
            "symbol": order.symbol,
        }

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        if not self.connected:
            return []
        if self._bridge_mode():
            try:
                result = self._bridge_call("fetch_open_orders", {"symbol": symbol})
            except Exception:
                return []
            if isinstance(result, list):
                return [item for item in result if isinstance(item, dict)]
            return []

        if symbol:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()

        if orders is None:
            return []

        result = []
        for order in orders:
            result.append(
                {
                    "id": str(order.ticket),
                    "symbol": order.symbol,
                    "type": "buy" if order.type == mt5.ORDER_TYPE_BUY else "sell",
                    "side": "buy"
                    if order.type
                    in [
                        mt5.ORDER_TYPE_BUY,
                        mt5.ORDER_TYPE_BUY_LIMIT,
                        mt5.ORDER_TYPE_BUY_STOP,
                    ]
                    else "sell",
                    "price": order.price_open,
                    "amount": order.volume_initial,
                    "filled": order.volume_current,
                    "filled_qty": order.volume_initial - order.volume_current,
                    "status": "open",
                    "info": order._asdict(),
                }
            )
        return result

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        _ = symbol
        if not self.connected:
            return False
        if self._bridge_mode():
            try:
                result = self._bridge_call(
                    "cancel_order", {"order_id": str(order_id), "symbol": symbol}
                )
                return bool(result)
            except Exception:
                return False

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order_id),
            "comment": "LuminaQuant Cancel",
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Cancel failed: {result.comment}")
            return False
        return True

    def __del__(self):
        if mt5:
            mt5.shutdown()
