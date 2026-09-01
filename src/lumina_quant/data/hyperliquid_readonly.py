"""Read-only Hyperliquid public-info helpers.

The helpers in this module intentionally expose only public market-data reads.
They are used to onboard Hyperliquid as a feature/regime source without adding
an execution driver or any trading surface.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from lumina_quant.utils.numeric import safe_float

DEFAULT_INFO_URL = "https://api.hyperliquid.xyz/info"


def _safe_float(value: Any) -> float | None:
    return safe_float(value, finite_only=True)


def _coin_from_symbol(symbol: str) -> str:
    token = str(symbol or "").strip().upper()
    if "/" in token:
        token = token.split("/", 1)[0]
    for suffix in ("USDT", "USDC", "USD"):
        if token.endswith(suffix) and len(token) > len(suffix):
            token = token[: -len(suffix)]
            break
    return token


def _symbol_from_coin(coin: str) -> str:
    return f"{str(coin).strip().upper()}/USDT"


@dataclass(frozen=True, slots=True)
class HyperliquidFundingPage:
    """One parsed `fundingHistory` response page."""

    coin: str
    rows: list[dict[str, Any]]
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None


class HyperliquidInfoClient:
    """Minimal JSON client for Hyperliquid's public `/info` endpoint."""

    def __init__(self, *, base_url: str = DEFAULT_INFO_URL, timeout_seconds: float = 30.0) -> None:
        self.base_url = str(base_url or DEFAULT_INFO_URL)
        self.timeout_seconds = float(timeout_seconds)

    def post(self, payload: dict[str, Any]) -> Any:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except urllib.error.URLError as exc:  # pragma: no cover - exercised by integration runs
            raise RuntimeError(f"Hyperliquid info request failed: {exc}") from exc
        return json.loads(body.decode("utf-8"))

    def meta_and_asset_contexts(self) -> Any:
        return self.post({"type": "metaAndAssetCtxs"})

    def funding_history(
        self,
        *,
        coin: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> Any:
        return self.post(
            {
                "type": "fundingHistory",
                "coin": str(coin).strip().upper(),
                "startTime": int(start_time_ms),
                "endTime": int(end_time_ms),
            }
        )

    def candle_snapshot(
        self,
        *,
        coin: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> Any:
        return self.post(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": str(coin).strip().upper(),
                    "interval": str(interval),
                    "startTime": int(start_time_ms),
                    "endTime": int(end_time_ms),
                },
            }
        )


def parse_meta_asset_context_rows(
    payload: Any,
    *,
    symbols: Iterable[str],
    timestamp_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Map `metaAndAssetCtxs` rows into futures-feature-point compatible rows."""
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    meta, contexts = payload[0], payload[1]
    universe = meta.get("universe") if isinstance(meta, dict) else None
    if not isinstance(universe, list) or not isinstance(contexts, list):
        return []

    wanted = {_coin_from_symbol(symbol) for symbol in symbols}
    collected_at = int(timestamp_ms if timestamp_ms is not None else time.time() * 1000)
    out: list[dict[str, Any]] = []
    for asset, context in zip(universe, contexts, strict=False):
        if not isinstance(asset, dict) or not isinstance(context, dict):
            continue
        coin = str(asset.get("name") or "").strip().upper()
        if coin not in wanted:
            continue
        mark_price = _safe_float(context.get("markPx"))
        oracle_price = _safe_float(context.get("oraclePx"))
        current_funding = _safe_float(context.get("funding"))
        open_interest = _safe_float(context.get("openInterest"))
        row = {
            "symbol": _symbol_from_coin(coin),
            "coin": coin,
            "timestamp_ms": collected_at,
            "funding_rate": current_funding,
            "funding_mark_price": mark_price,
            "mark_price": mark_price,
            "index_price": oracle_price,
            "open_interest": open_interest,
            "raw_context": dict(context),
            "raw_asset": dict(asset),
        }
        out.append(row)
    return out


def parse_funding_history_page(coin: str, payload: Any) -> HyperliquidFundingPage:
    """Parse a `fundingHistory` response into feature-point rows."""
    rows: list[dict[str, Any]] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        timestamp = item.get("time")
        try:
            timestamp_ms = int(timestamp)
        except Exception:
            continue
        funding_rate = _safe_float(item.get("fundingRate"))
        if funding_rate is None:
            continue
        rows.append(
            {
                "symbol": _symbol_from_coin(coin),
                "coin": str(coin).strip().upper(),
                "timestamp_ms": timestamp_ms,
                "funding_rate": funding_rate,
                "raw_premium": _safe_float(item.get("premium")),
            }
        )
    rows.sort(key=lambda row: int(row["timestamp_ms"]))
    return HyperliquidFundingPage(
        coin=str(coin).strip().upper(),
        rows=rows,
        first_timestamp_ms=int(rows[0]["timestamp_ms"]) if rows else None,
        last_timestamp_ms=int(rows[-1]["timestamp_ms"]) if rows else None,
    )


def parse_candle_snapshot(payload: Any) -> list[dict[str, Any]]:
    """Parse candleSnapshot rows without treating them as raw-first data."""
    rows: list[dict[str, Any]] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            start_ms = int(item.get("t"))
            end_ms = int(item.get("T"))
        except Exception:
            continue
        row = {
            "coin": str(item.get("s") or "").strip().upper(),
            "interval": str(item.get("i") or ""),
            "timestamp_ms": start_ms,
            "end_timestamp_ms": end_ms,
            "open": _safe_float(item.get("o")),
            "high": _safe_float(item.get("h")),
            "low": _safe_float(item.get("l")),
            "close": _safe_float(item.get("c")),
            "volume": _safe_float(item.get("v")),
        }
        rows.append(row)
    rows.sort(key=lambda row: int(row["timestamp_ms"]))
    return rows


__all__ = [
    "DEFAULT_INFO_URL",
    "HyperliquidFundingPage",
    "HyperliquidInfoClient",
    "parse_candle_snapshot",
    "parse_funding_history_page",
    "parse_meta_asset_context_rows",
]
