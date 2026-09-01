"""Compatibility wrapper for volatility-compression VWAP reversion strategy."""

from __future__ import annotations

from lumina_quant.strategies.candidate_vol_compression_reversion import (
    VolatilityCompressionReversionStrategy,
)


class VolCompressionVWAPReversionStrategy(VolatilityCompressionReversionStrategy):
    """Alias strategy with the explicit advanced-pipeline class name."""

    pass


class VolCompressionVwapReversionStrategy(VolCompressionVWAPReversionStrategy):
    """Backwards-compatible alias (legacy capitalization)."""

    pass
