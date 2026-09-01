"""Wiring guard for the 2026-08-20 alpha-sleeve batch (W2 regression).

``resolve_params_from_schema`` SILENTLY DROPS unknown params, so a candidate
row whose ``params`` drift from the class schema would run with defaults and
nobody would notice — this test is the guard.  It builds the default manifest
rows over a >=12-name crypto universe plus the offsession TradFi universe on
``['1h', '4h', '1d']`` and asserts, per batch class:

- the exact per-class row count (a builder gate silently dropping cells, or a
  slice edit adding cells, flips the count);
- every row's class resolves through the strategy registry;
- every row constructs against a minimal bars stub + ``SimpleQueue`` with the
  row's own params and survives a ``get_state``/``set_state`` roundtrip;
- every row's params are contained in ``cls.get_param_schema()`` — the ONLY
  allowed exception is the offsession ``tradfi_symbols`` universe tuple, which
  the constructor consumes explicitly BEFORE schema resolution.

No backtest is run; no data files are read; everything is deterministic.
"""

from __future__ import annotations

from collections import Counter
from functools import cache
from queue import SimpleQueue
from typing import Any

from lumina_quant.strategies.offsession_basis_dislocation_alpha_sleeves import (
    _DEFAULT_TRADFI_UNIVERSE,
)
from lumina_quant.strategies.registry import resolve_strategy_class
from lumina_quant.strategy_factory import build_binance_futures_candidates

_TIMEFRAMES = ("1h", "4h", "1d")

# Twelve crypto perps (none tradfi, none metals): the guarded_ls taker-flow
# cell requires min_symbols == 12, so this is the smallest universe that
# admits EVERY pre-registered cell of the batch.
_CRYPTO_UNIVERSE = (
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "TON/USDT",
    "LINK/USDT",
    "LTC/USDT",
)

# Pre-registered default-manifest row counts per batch class (variants x
# timeframes surviving each builder's min_symbols gates on this universe).
_EXPECTED_ROWS_PER_CLASS = {
    "CrossSectionalResidualTakerFlowStrategy": 4,
    "BasisFundingGapConvergenceStrategy": 3,
    "OffSessionBasisDislocationStrategy": 2,
    "SalienceTheoryValueStrategy": 6,
    "ProspectTheoryValueStrategy": 6,
    "OpenInterestGrowthPressureStrategy": 2,
}

# Params legitimately OUTSIDE the class schema, consumed explicitly by the
# constructor before ``resolve_params_from_schema`` runs.  ONLY the offsession
# universe tuple is allowed; anything else here means silent param loss.
_SCHEMA_ALLOWLIST = {
    "OffSessionBasisDislocationStrategy": frozenset({"tradfi_symbols"}),
}


class _Bars:
    """Minimal bars stub: the constructors only read ``symbol_list``."""

    def __init__(self, symbols: tuple[str, ...]) -> None:
        self.symbol_list = list(symbols)


@cache
def _batch_rows() -> tuple[Any, ...]:
    rows = build_binance_futures_candidates(
        timeframes=list(_TIMEFRAMES),
        symbols=[*_CRYPTO_UNIVERSE, *_DEFAULT_TRADFI_UNIVERSE],
    )
    return tuple(row for row in rows if row.strategy_class in _EXPECTED_ROWS_PER_CLASS)


def test_batch_row_counts_per_class() -> None:
    counts = Counter(row.strategy_class for row in _batch_rows())
    assert dict(counts) == _EXPECTED_ROWS_PER_CLASS


def test_rows_resolve_construct_and_state_roundtrip() -> None:
    for row in _batch_rows():
        cls = resolve_strategy_class(row.strategy_class)
        strategy = cls(_Bars(row.symbols), SimpleQueue(), **row.params)
        state = strategy.get_state()
        strategy.set_state(state)
        assert strategy.get_state() == state, (row.name, "state roundtrip drift")


def test_row_params_are_contained_in_class_schema() -> None:
    # ``resolve_params_from_schema`` silently drops unknown params, so schema
    # containment is asserted here instead of trusted at construction time.
    for row in _batch_rows():
        cls = resolve_strategy_class(row.strategy_class)
        allowed_extra = _SCHEMA_ALLOWLIST.get(row.strategy_class, frozenset())
        extra = set(row.params) - set(cls.get_param_schema())
        assert extra <= allowed_extra, (row.name, sorted(extra))
        # The allowlisted key must actually be present on offsession rows so
        # the universe restriction cannot silently disappear from the slice.
        if row.strategy_class == "OffSessionBasisDislocationStrategy":
            assert "tradfi_symbols" in row.params, row.name
