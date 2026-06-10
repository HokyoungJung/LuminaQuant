"""Config-driven DataCollector — unified entry point for all data-collection paths.

Phase 3: ``DataCollector.from_runtime_config(rt)`` is the single factory.

Design:
- Driver-keyed adapter selection: binance_futures / mt5 / polymarket.
- Data-kind selection via ``data.kinds`` in the profile YAML:
  ohlcv / funding / feature_points / aggtrades_tick.
- ``aggtrades_tick`` is keyless read-only public market data; safety gate is
  ``data.kinds`` config (``real.yaml`` excludes it for always-on live).
  See AGENTS.md "tick collection safety axis" for rationale.
- Storage root follows ``storage.market_data_parquet_path`` (``DataConfig.tick_path``
  overrides it for raw aggTrades archives).
- MT5 and Polymarket are reachable through the same factory (driver-keyed), not
  special-cased outside it.
"""

from __future__ import annotations

import os
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lumina_quant.configuration.schema import DataConfig, RuntimeConfig

# ---------------------------------------------------------------------------
# Data-kind constants
# ---------------------------------------------------------------------------

DATA_KIND_OHLCV = "ohlcv"
DATA_KIND_FUNDING = "funding"
DATA_KIND_FEATURE_POINTS = "feature_points"
DATA_KIND_AGGTRADES_TICK = "aggtrades_tick"

_VALID_DATA_KINDS: frozenset[str] = frozenset(
    {DATA_KIND_OHLCV, DATA_KIND_FUNDING, DATA_KIND_FEATURE_POINTS, DATA_KIND_AGGTRADES_TICK}
)
_VALID_DRIVERS: frozenset[str] = frozenset({"binance_futures", "mt5", "polymarket"})


# ---------------------------------------------------------------------------
# DataCollectorConfig
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DataCollectorConfig:
    """Fully-resolved data-collection settings derived from RuntimeConfig.

    Instantiate via ``DataCollectorConfig.from_runtime_config(rt)`` — do not
    construct manually in application code.
    """

    exchange: str = "binance"
    """Logical exchange token (``storage.market_data_exchange``)."""

    driver: str = "binance_futures"
    """Exchange driver identifier (``live.exchange.driver``).
    Controls which adapter class is built by the factory.
    """

    parquet_root: Path = field(default_factory=lambda: Path("data/market_parquet"))
    """Primary parquet storage root (``storage.market_data_parquet_path``)."""

    symbols: list[str] = field(default_factory=list)
    """Symbol universe (``trading.symbols``)."""

    is_testnet: bool = True
    """True when operating in testnet/paper/replay mode.
    Derived from ``live.testnet`` or ``live.mode == "paper"``.
    """

    kinds: list[str] = field(default_factory=lambda: list(DataConfig().kinds))
    """Active data kinds (``data.kinds``).
    Subset of: ohlcv, funding, feature_points, aggtrades_tick.
    """

    tick_path: Path | None = None
    """Override storage root for raw aggTrades archives (``data.tick_path``).
    None → use ``parquet_root``.
    """

    api_key: str = ""
    """Exchange API key (``live.api_key``).  Testnet/paper keys only — NEVER real-order keys.
    """

    secret_key: str = ""
    """Exchange secret key (``live.secret_key``).  Testnet/paper only."""

    market_type: str = "future"
    """Market type passed to the exchange adapter (``live.exchange.market_type``)."""

    @classmethod
    def from_runtime_config(cls, rt: RuntimeConfig) -> DataCollectorConfig:
        """Derive DataCollectorConfig from a fully-loaded RuntimeConfig."""
        is_testnet: bool
        if rt.live.testnet is not None:
            is_testnet = bool(rt.live.testnet)
        else:
            is_testnet = str(rt.live.mode or "paper").strip().lower() in {"paper", "testnet"}

        tick_path_raw = str(getattr(rt.data, "tick_path", "") or "").strip()
        return cls(
            exchange=str(rt.storage.market_data_exchange or "binance"),
            driver=str(rt.live.exchange.driver or "binance_futures"),
            parquet_root=Path(rt.storage.market_data_parquet_path or "data/market_parquet"),
            symbols=list(rt.trading.symbols),
            is_testnet=is_testnet,
            kinds=list(rt.data.kinds),
            tick_path=Path(tick_path_raw) if tick_path_raw else None,
            api_key=str(rt.live.api_key or ""),
            secret_key=str(rt.live.secret_key or ""),
            market_type=str(rt.live.exchange.market_type or "future"),
        )


# ---------------------------------------------------------------------------
# DataCollector
# ---------------------------------------------------------------------------


class DataCollector:
    """Config-driven collector — single entry point for all data paths.

    Instantiate via ``DataCollector.from_runtime_config(rt)``.

    Example::

        from lumina_quant.configuration import get_default_runtime_config
        from lumina_quant.data.collector import DataCollector

        rt = get_default_runtime_config()
        collector = DataCollector.from_runtime_config(rt)
        # Exchange + symbols + data-kinds all come from the YAML profile;
        # no code change needed to switch exchange or symbol universe.
    """

    def __init__(
        self,
        cfg: DataCollectorConfig,
        *,
        _exchange_client: Any | None = None,
    ) -> None:
        self._cfg = cfg
        self._exchange_client = _exchange_client

    @classmethod
    def from_runtime_config(cls, rt: RuntimeConfig) -> DataCollector:
        """Build DataCollector from RuntimeConfig.

        Driver-keyed adapter selection::

            live.exchange.driver = "binance_futures"  → BinanceFuturesRESTClient
            live.exchange.driver = "mt5"              → MT5Exchange
            live.exchange.driver = "polymarket"       → PolymarketExchange
        """
        cfg = DataCollectorConfig.from_runtime_config(rt)
        exchange_client = _build_exchange_client(cfg, rt=rt)
        return cls(cfg, _exchange_client=exchange_client)

    @property
    def config(self) -> DataCollectorConfig:
        """Resolved configuration (read-only)."""
        return self._cfg

    @property
    def repo(self) -> Any:
        """``ParquetMarketDataRepository`` for this collector's storage root.

        The repository root comes from ``storage.market_data_parquet_path`` via
        ``DataCollectorConfig.parquet_root`` — no manual path threading required.
        """
        from lumina_quant.storage.parquet import ParquetMarketDataRepository

        return ParquetMarketDataRepository(str(self._cfg.parquet_root))

    # ------------------------------------------------------------------
    # Data-collection methods (one per data kind)
    # ------------------------------------------------------------------

    def ensure_ohlcv_coverage(
        self,
        *,
        symbols: list[str] | None = None,
        timeframe: str = "1m",
        since_ms: int | None = None,
        until_ms: int | None = None,
        force_full: bool = False,
    ) -> list[dict[str, Any]]:
        """Ensure OHLCV coverage exists in parquet storage.

        Maps to ``DATA_KIND_OHLCV``.
        """
        from lumina_quant.data_sync import ensure_market_data_coverage

        symbol_list = symbols if symbols is not None else self._cfg.symbols
        stats = ensure_market_data_coverage(
            exchange=self._exchange_client,
            db_path=str(self._cfg.parquet_root),
            exchange_id=self._cfg.exchange,
            symbol_list=symbol_list,
            timeframe=timeframe,
            since_ms=since_ms,
            until_ms=until_ms,
            force_full=bool(force_full),
        )
        return [
            {
                "symbol": item.symbol,
                "fetched_rows": int(item.fetched_rows),
                "upserted_rows": int(item.upserted_rows),
                "first_timestamp_ms": item.first_timestamp_ms,
                "last_timestamp_ms": item.last_timestamp_ms,
            }
            for item in stats
        ]

    def collect_feature_points(
        self,
        *,
        symbols: list[str] | None = None,
        since_ms: int,
        until_ms: int | None = None,
        feature_profile: str = "full",
        execute: bool = True,
    ) -> dict[str, Any]:
        """Collect funding rates and feature points.

        Maps to ``DATA_KIND_FUNDING`` and ``DATA_KIND_FEATURE_POINTS``.
        """
        from datetime import UTC, datetime

        from lumina_quant.data_collector import collect_strategy_support_data

        symbol_list = symbols if symbols is not None else self._cfg.symbols
        effective_until = until_ms or int(datetime.now(UTC).timestamp() * 1000)
        return collect_strategy_support_data(
            db_path=str(self._cfg.parquet_root),
            exchange_id=self._cfg.exchange,
            symbol_list=symbol_list,
            since=since_ms,
            until=effective_until,
            feature_profile=feature_profile,
            execute=execute,
        )

    def collect_aggtrades_raw(
        self,
        *,
        symbol: str,
        since_ms: int | None = None,
        until_ms: int | None = None,
        bootstrap_lookback_hours: int = 24,
    ) -> dict[str, Any]:
        """Collect raw aggTrades canonical source with checkpoint resume.

        Maps to ``DATA_KIND_AGGTRADES_TICK``.

        Public read-only market data — no order side effects.  The safety gate
        is the ``data.kinds`` config knob: ``real.yaml`` excludes
        ``aggtrades_tick`` so the always-on live process never runs this path.
        Testnet vs prod is irrelevant here — the Phase 0 aggTrades fixture was
        captured from the prod public REST endpoint (fapi.binance.com, keyless).
        """
        from lumina_quant.data_collector import collect_binance_aggtrades_raw

        tick_db = str(self._cfg.tick_path or self._cfg.parquet_root)
        return collect_binance_aggtrades_raw(
            db_path=tick_db,
            exchange_id=self._cfg.exchange,
            symbol=symbol,
            market_type=self._cfg.market_type,
            api_key=self._cfg.api_key,
            secret_key=self._cfg.secret_key,
            testnet=self._cfg.is_testnet,
            since_ms=since_ms,
            until_ms=until_ms,
            bootstrap_lookback_hours=bootstrap_lookback_hours,
        )

    def close(self) -> None:
        """Release underlying exchange client resources."""
        if self._exchange_client is not None:
            close_fn = getattr(self._exchange_client, "close", None)
            if callable(close_fn):
                close_fn()

    def __enter__(self) -> DataCollector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Driver-keyed exchange client factory (internal)
# ---------------------------------------------------------------------------


def _build_exchange_client(cfg: DataCollectorConfig, *, rt: RuntimeConfig) -> Any:
    """Build an exchange client from the resolved collector config.

    Dispatch is driver-keyed — no special-casing outside this function.
    All paths respect testnet/paper/replay safety constraints:
    NEVER real API keys for order placement.
    """
    driver = str(cfg.driver or "binance_futures").strip().lower()

    if driver in {"binance_futures", "binance_native", "binance", ""}:
        return _build_binance_futures_client(cfg)

    if driver == "mt5":
        return _build_mt5_client(rt)

    if driver == "polymarket":
        return _build_polymarket_client(rt)

    raise ValueError(
        f"Unknown exchange driver: {driver!r}. Supported: binance_futures, mt5, polymarket."
    )


def _build_binance_futures_client(cfg: DataCollectorConfig) -> Any:
    from lumina_quant.exchanges.binance_futures_client import (
        BinanceFuturesClientConfig,
        BinanceFuturesRESTClient,
    )

    client_cfg = BinanceFuturesClientConfig(
        api_key=cfg.api_key,
        secret_key=cfg.secret_key,
        testnet=cfg.is_testnet,
    )
    return BinanceFuturesRESTClient(client_cfg)


def _build_mt5_client(rt: RuntimeConfig) -> Any:
    from lumina_quant.exchanges.mt5_exchange import MT5Exchange

    mt5_cfg = types.SimpleNamespace(
        MT5_BRIDGE_PYTHON=str(rt.live.mt5_bridge_python or ""),
        MT5_BRIDGE_SCRIPT=str(rt.live.mt5_bridge_script or ""),
        MT5_MAGIC=int(rt.live.mt5_magic),
        MT5_DEVIATION=int(rt.live.mt5_deviation),
        MT5_BRIDGE_USE_WSLPATH=bool(rt.live.mt5_bridge_use_wslpath),
    )
    return MT5Exchange(mt5_cfg)


def _build_polymarket_client(rt: RuntimeConfig) -> Any:
    from lumina_quant.exchanges.polymarket_exchange import PolymarketExchange

    poly = rt.live.polymarket
    poly_cfg = types.SimpleNamespace(
        POLYMARKET_HOST=str(poly.host or "https://clob.polymarket.com"),
        POLYMARKET_GAMMA_HOST=str(poly.gamma_host or "https://gamma-api.polymarket.com"),
        POLYMARKET_DATA_HOST=str(poly.data_host or "https://data-api.polymarket.com"),
        POLYMARKET_MARKET_WS_URL=str(
            poly.market_ws_url or "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        ),
        POLYMARKET_USER_WS_URL=str(
            poly.user_ws_url or "wss://ws-subscriptions-clob.polymarket.com/ws/user"
        ),
        POLYMARKET_CHAIN_ID=int(poly.chain_id),
        POLYMARKET_PRIVATE_KEY=os.environ.get(str(poly.private_key_env or ""), ""),
        POLYMARKET_API_KEY=os.environ.get(str(poly.api_key_env or ""), ""),
        POLYMARKET_API_SECRET=os.environ.get(str(poly.api_secret_env or ""), ""),
        POLYMARKET_API_PASSPHRASE=os.environ.get(str(poly.api_passphrase_env or ""), ""),
        POLYMARKET_FUNDER=str(poly.funder or ""),
        POLYMARKET_SIGNATURE_TYPE=int(poly.signature_type),
        ALLOW_REAL_EXECUTION=bool(poly.allow_real_execution),
    )
    return PolymarketExchange(poly_cfg)


__all__ = [
    "DATA_KIND_AGGTRADES_TICK",
    "DATA_KIND_FEATURE_POINTS",
    "DATA_KIND_FUNDING",
    "DATA_KIND_OHLCV",
    "DataCollector",
    "DataCollectorConfig",
]
