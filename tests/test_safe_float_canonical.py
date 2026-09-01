"""Tests for the canonical ``safe_float`` numeric coercion helper."""

from __future__ import annotations

import math

import pytest

from lumina_quant.utils.numeric import safe_float


def test_valid_int() -> None:
    result = safe_float(5)
    assert result == 5.0
    assert isinstance(result, float)


def test_valid_float() -> None:
    assert safe_float(3.14) == 3.14


def test_valid_str_number() -> None:
    assert safe_float("2.5") == 2.5
    assert safe_float("-7") == -7.0


@pytest.mark.parametrize("bad", [None, "", "abc", object()])
def test_invalid_returns_default_none(bad: object) -> None:
    assert safe_float(bad) is None


@pytest.mark.parametrize("bad", [None, "", "abc", object()])
def test_invalid_returns_default_zero(bad: object) -> None:
    assert safe_float(bad, 0.0) == 0.0


def test_inf_finite_only_true_returns_default() -> None:
    assert safe_float(float("inf"), 0.0, finite_only=True) == 0.0
    assert safe_float(float("-inf"), 0.0, finite_only=True) == 0.0
    assert safe_float(float("inf"), finite_only=True) is None


def test_nan_finite_only_true_returns_default() -> None:
    assert safe_float(float("nan"), 0.0, finite_only=True) == 0.0
    assert safe_float(float("nan"), finite_only=True) is None


def test_inf_finite_only_false_passes_through() -> None:
    assert safe_float(float("inf")) == float("inf")
    assert safe_float(float("-inf")) == float("-inf")


def test_nan_finite_only_false_passes_through() -> None:
    assert math.isnan(safe_float(float("nan")))


def test_default_propagation_custom_sentinel() -> None:
    sentinel = -1_000_000.0
    assert safe_float("abc", sentinel) == sentinel
    assert safe_float(float("nan"), sentinel, finite_only=True) == sentinel


def test_string_inf_finite_only() -> None:
    # ``float("inf")`` parses, so finite_only must still reject it.
    assert safe_float("inf", 0.0, finite_only=True) == 0.0
    assert safe_float("inf") == float("inf")
