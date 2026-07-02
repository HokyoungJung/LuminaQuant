"""Data-only invariants for the forex parameter seam (no execution logic)."""

from __future__ import annotations

import pytest

from lumina_quant.research import forex_params as fx


def test_registry_pip_sizes_are_consistent() -> None:
    for spec in fx.DEFAULT_FOREX_PAIR_SPECS:
        assert spec.pip_size == pytest.approx(10.0 ** (-spec.pip_decimal_place))
        assert spec.contract_size > 0.0


def test_jpy_pairs_use_two_decimal_pips() -> None:
    spec = fx.get_forex_pair_spec("USD/JPY")
    assert spec.pip_decimal_place == 2
    assert spec.pip_size == pytest.approx(0.01)


def test_non_jpy_majors_use_four_decimal_pips() -> None:
    spec = fx.get_forex_pair_spec("EUR/USD")
    assert spec.pip_decimal_place == 4
    assert spec.pip_size == pytest.approx(0.0001)


def test_registry_lookup_is_case_insensitive() -> None:
    assert fx.get_forex_pair_spec("eur/usd").symbol == "EUR/USD"


def test_unknown_pair_fails_loud() -> None:
    with pytest.raises(KeyError):
        fx.get_forex_pair_spec("XXX/YYY")


def test_registry_helper_returns_symbol_keyed_copy() -> None:
    registry = fx.forex_pair_registry()
    assert set(registry) == {spec.symbol for spec in fx.DEFAULT_FOREX_PAIR_SPECS}
    assert registry["GBP/USD"].base_currency == "GBP"
