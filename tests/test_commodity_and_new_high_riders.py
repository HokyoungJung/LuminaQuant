"""Candidate-wiring tests for the commodity managed-futures + 52-week-high riders.

These are deterministic wiring tests over a mixed universe (crypto + the 8
commodity perps + equity perps). They assert ROUTING only — the underlying rider
strategy behavior is already covered by ``tests/test_return_rider_alpha_sleeves``
because all four builders REUSE the proven return-rider classes:

- the three commodity builders emit single-asset candidates ONLY for the 8
  commodity symbols (XAU/XAG/XPT/XPD + COPPER/CL/BZ/NATGAS), long AND short;
- the 52-week-high builder emits single-asset candidates ONLY for equity symbols,
  long-only;
- every new candidate is ``candidate_mix_type=="single"`` on a >=30m timeframe
  ({30m, 1h, 4h, 1d}); and
- NO crypto symbol leaks into any of these new builders' outputs.
"""

from __future__ import annotations

import unittest

from lumina_quant.research_universe import (
    BINANCE_TRADFI_COMMODITY_SYMBOLS,
    BINANCE_TRADFI_EQUITY_SYMBOLS,
)
from lumina_quant.strategy_factory.candidate_library import (
    _COMMODITY_TREND_UNIVERSE,
    build_binance_futures_candidates,
)
from lumina_quant.strategy_factory.selection import candidate_mix_type
from lumina_quant.symbols import canonicalize_symbol_list

# Crypto symbols that must NEVER appear in any of the new commodity/52w-high
# builders' outputs (they are routed through the dedicated crypto riders).
_CRYPTO_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")
# A representative slice of the equity universe to exercise the 52w-high builder.
_EQUITY_SAMPLE = ("TSLA/USDT", "NVDA/USDT", "AAPL/USDT", "MSFT/USDT")
_ALLOWED_TIMEFRAMES = {"30m", "1h", "4h", "1d"}

_COMMODITY_PREFIXES = (
    "commodity_adaptive_trend_rider",
    "commodity_breakout_rider",
    "commodity_acceleration_rider",
)
_NEW_HIGH_PREFIX = "equity_new_high_breakout_rider"


def _build_mixed_rows():
    universe = [
        *_CRYPTO_SYMBOLS,
        *canonicalize_symbol_list(BINANCE_TRADFI_COMMODITY_SYMBOLS),
        *_EQUITY_SAMPLE,
    ]
    return build_binance_futures_candidates(
        timeframes=["30m", "1h", "4h", "1d"],
        symbols=universe,
    )


class TestCommodityAndNewHighRiders(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _build_mixed_rows()
        cls.commodity_canon = set(canonicalize_symbol_list(BINANCE_TRADFI_COMMODITY_SYMBOLS))
        cls.equity_canon = set(canonicalize_symbol_list(BINANCE_TRADFI_EQUITY_SYMBOLS))
        cls.crypto_canon = set(canonicalize_symbol_list(_CRYPTO_SYMBOLS))

    def _rows_with_prefix(self, prefix: str):
        return [row for row in self.rows if row.name.startswith(prefix)]

    def test_commodity_universe_constant_matches_research_universe(self) -> None:
        # Constant 1: _COMMODITY_TREND_UNIVERSE is the 8-symbol commodity set.
        self.assertEqual(
            tuple(_COMMODITY_TREND_UNIVERSE),
            tuple(dict.fromkeys(BINANCE_TRADFI_COMMODITY_SYMBOLS)),
        )
        self.assertEqual(len(_COMMODITY_TREND_UNIVERSE), 8)

    def test_commodity_riders_route_only_to_eight_commodity_symbols(self) -> None:
        for prefix in _COMMODITY_PREFIXES:
            rows = self._rows_with_prefix(prefix)
            self.assertTrue(rows, f"expected {prefix} candidates over the mixed universe")
            symbols = {symbol for row in rows for symbol in row.symbols}
            # Exactly the 8 commodity perps — no more, no fewer.
            self.assertEqual(
                symbols,
                self.commodity_canon,
                f"{prefix} must route to all and only the 8 commodity symbols",
            )

    def test_commodity_riders_single_asset_allowed_tfs_and_long_short(self) -> None:
        for prefix in _COMMODITY_PREFIXES:
            for row in self._rows_with_prefix(prefix):
                self.assertEqual(len(row.symbols), 1, f"{prefix} must be single-asset")
                self.assertEqual(candidate_mix_type(row.to_dict()), "single")
                self.assertIn(row.timeframe, _ALLOWED_TIMEFRAMES)
                self.assertIn(row.timeframe, {"4h", "1d"})  # commodity riders: 4h + 1d
                # Commodities trend BOTH ways: allow_short must be True.
                self.assertTrue(bool(row.metadata.get("allow_short")))
                self.assertIs(row.params["allow_short"], True)
                self.assertIn("commodity", row.tags)
                self.assertIn("single_asset", row.tags)
                self.assertIn("return_rider", row.tags)

    def test_new_high_rider_routes_only_to_equity_symbols(self) -> None:
        rows = self._rows_with_prefix(_NEW_HIGH_PREFIX)
        self.assertTrue(rows, "expected 52-week-high rider candidates over the mixed universe")
        symbols = {symbol for row in rows for symbol in row.symbols}
        # Every emitted symbol is an equity perp; none are commodity/crypto.
        self.assertTrue(symbols.issubset(self.equity_canon))
        self.assertEqual(symbols, set(canonicalize_symbol_list(_EQUITY_SAMPLE)))
        self.assertFalse(symbols & self.commodity_canon)
        self.assertFalse(symbols & self.crypto_canon)

    def test_new_high_rider_single_asset_long_only_and_252_lookback(self) -> None:
        rows = self._rows_with_prefix(_NEW_HIGH_PREFIX)
        for row in rows:
            self.assertEqual(len(row.symbols), 1, "52w-high rider must be single-asset")
            self.assertEqual(candidate_mix_type(row.to_dict()), "single")
            self.assertIn(row.timeframe, _ALLOWED_TIMEFRAMES)
            self.assertIn(row.timeframe, {"4h", "1d"})  # 1d primary + 4h
            # 52-week-high momentum is a LONG-only effect.
            self.assertFalse(bool(row.metadata.get("allow_short")))
            self.assertIs(row.params["allow_short"], False)
            # ~252-bar (52-week) new-high Donchian lookback (schema max 4096).
            self.assertEqual(int(row.params["donchian_window"]), 252)
            self.assertEqual(row.strategy_class, "VolatilityBreakoutRiderStrategy")
            for tag in ("breakout", "momentum", "52w_high", "single_asset", "equity"):
                self.assertIn(tag, row.tags)

    def test_no_crypto_symbol_leaks_into_new_builders(self) -> None:
        for prefix in (*_COMMODITY_PREFIXES, _NEW_HIGH_PREFIX):
            for row in self._rows_with_prefix(prefix):
                for symbol in row.symbols:
                    self.assertNotIn(
                        symbol,
                        self.crypto_canon,
                        f"{prefix} leaked crypto symbol {symbol}",
                    )

    def test_all_new_builders_emit_timeframes_within_30m_floor(self) -> None:
        # No sub-30m timeframe (1s/1m/5m/15m) is ever wired for these riders.
        for prefix in (*_COMMODITY_PREFIXES, _NEW_HIGH_PREFIX):
            tfs = {row.timeframe for row in self._rows_with_prefix(prefix)}
            self.assertTrue(tfs.issubset(_ALLOWED_TIMEFRAMES), f"{prefix} tfs={tfs}")


if __name__ == "__main__":
    unittest.main()
