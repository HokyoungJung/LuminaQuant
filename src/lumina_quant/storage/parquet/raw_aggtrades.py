"""Raw aggTrades layout helpers.

This module intentionally re-exports the repository implementation while the
raw aggTrades API is incrementally modularized.
"""

from lumina_quant.storage.parquet.ohlcv_repo import ParquetMarketDataRepository

__all__ = ["ParquetMarketDataRepository"]
