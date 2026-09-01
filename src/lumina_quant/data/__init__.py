"""Data-domain façade modules."""

from lumina_quant.data.collector import DataCollector, DataCollectorConfig
from lumina_quant.data.ohlcv import load_data_dict_from_db, load_data_dict_from_parquet
from lumina_quant.data.symbols import canonical_symbol
from lumina_quant.data.timeframe import normalize_timeframe_token, timeframe_to_milliseconds
from lumina_quant.data.symbol_lifecycle import (
    build_fold_membership_manifest,
    build_symbol_lifecycle_registry,
    is_symbol_active,
    load_symbol_lifecycle_registry,
    validate_fold_membership_manifest,
    validate_symbol_lifecycle_registry,
)

__all__ = [
    "DataCollector",
    "DataCollectorConfig",
    "build_fold_membership_manifest",
    "build_symbol_lifecycle_registry",
    "canonical_symbol",
    "is_symbol_active",
    "load_data_dict_from_db",
    "load_data_dict_from_parquet",
    "load_symbol_lifecycle_registry",
    "normalize_timeframe_token",
    "timeframe_to_milliseconds",
    "validate_fold_membership_manifest",
    "validate_symbol_lifecycle_registry",
]
