"""OHLCV loader façade for backward-compatible access."""

from lumina_quant.market_data import load_data_dict_from_db, load_data_dict_from_parquet

__all__ = ["load_data_dict_from_db", "load_data_dict_from_parquet"]
