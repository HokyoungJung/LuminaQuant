"""Materialized-window manifest helpers.

This module intentionally re-exports the repository implementation while the
materialization API is incrementally modularized.
"""

from lumina_quant.storage.parquet.ohlcv_repo import ParquetMarketDataRepository

__all__ = ["ParquetMarketDataRepository"]
