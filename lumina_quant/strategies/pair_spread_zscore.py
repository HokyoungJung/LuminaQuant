"""Pair spread z-score strategy wrapper with deterministic default pair anchors."""

from __future__ import annotations

from collections.abc import Sequence

from lumina_quant.strategies.pair_trading_zscore import PairTradingZScoreStrategy
from lumina_quant.symbols import canonical_symbol

_DEFAULT_PAIR_SET: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "ETH/USDT"),
    ("ETH/USDT", "SOL/USDT"),
    ("XAU/USDT", "XAG/USDT"),
)


class PairSpreadZScoreStrategy(PairTradingZScoreStrategy):
    """Alias wrapper around :class:`PairTradingZScoreStrategy` with stable defaults."""

    @staticmethod
    def _resolve_default_pair(symbol_list: Sequence[str]) -> tuple[str, str]:
        canonical = [canonical_symbol(symbol) for symbol in symbol_list if str(symbol).strip()]
        canonical = [symbol for symbol in canonical if symbol]
        universe = set(canonical)

        for left, right in _DEFAULT_PAIR_SET:
            if left in universe and right in universe:
                return left, right

        if len(canonical) >= 2:
            ordered = sorted(dict.fromkeys(canonical))
            return ordered[0], ordered[1]

        raise ValueError("PairSpreadZScoreStrategy requires at least two symbols.")

    def __init__(self, bars, events, *args, symbol_x=None, symbol_y=None, **kwargs):
        if not symbol_x or not symbol_y:
            left, right = self._resolve_default_pair(getattr(bars, "symbol_list", []))
            symbol_x = symbol_x or left
            symbol_y = symbol_y or right
        super().__init__(bars, events, *args, symbol_x=symbol_x, symbol_y=symbol_y, **kwargs)
