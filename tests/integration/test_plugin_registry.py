"""Integration tests for the Phase 1 plugin registry.

Covers all three required plugin kinds: strategy, indicator, portfolio.
Each test registers a local test class and verifies round-trip behaviour
through the global GLOBAL_REGISTRY accessor.
"""

from __future__ import annotations

import pytest

from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY, register


# ---------------------------------------------------------------------------
# Helpers — isolated registry per test to avoid cross-test pollution
# ---------------------------------------------------------------------------

def _register_and_lookup(kind: str, name: str, interface: str | None = None):
    """Register a fresh class, return (cls, retrieved_cls, interface_tag)."""

    @register(kind, name, interface=interface)
    class _TestPlugin:
        pass

    retrieved = GLOBAL_REGISTRY.get(kind, name)
    tag = GLOBAL_REGISTRY.get_interface(kind, name)
    return _TestPlugin, retrieved, tag


# ---------------------------------------------------------------------------
# 1. Strategy kind
# ---------------------------------------------------------------------------

class TestStrategyRegistration:
    def test_event_driven_strategy_registers_and_retrieves(self):
        cls, retrieved, tag = _register_and_lookup(
            "strategy", "_IntegTestEventDrivenStrategy", interface="event_driven"
        )
        assert retrieved is cls
        assert tag == "event_driven"
        assert "strategy" in GLOBAL_REGISTRY.list_kinds()
        assert "_IntegTestEventDrivenStrategy" in GLOBAL_REGISTRY.list_names("strategy")

    def test_polars_batch_strategy_registers_with_interface_tag(self):
        cls, retrieved, tag = _register_and_lookup(
            "strategy", "_IntegTestPolarsStrategy", interface="polars_batch"
        )
        assert retrieved is cls
        assert tag == "polars_batch"

    def test_strategy_get_all_returns_dict(self):
        @register("strategy", "_IntegTestGetAllStrategy")
        class _S:
            pass

        all_strats = GLOBAL_REGISTRY.get_all("strategy")
        assert isinstance(all_strats, dict)
        assert "_IntegTestGetAllStrategy" in all_strats
        assert all_strats["_IntegTestGetAllStrategy"] is _S


# ---------------------------------------------------------------------------
# 2. Indicator kind
# ---------------------------------------------------------------------------

class TestIndicatorRegistration:
    def test_indicator_registers_and_retrieves(self):
        cls, retrieved, tag = _register_and_lookup(
            "indicator", "_IntegTestRSIIndicator"
        )
        assert retrieved is cls
        assert tag is None
        assert "indicator" in GLOBAL_REGISTRY.list_kinds()
        assert "_IntegTestRSIIndicator" in GLOBAL_REGISTRY.list_names("indicator")

    def test_indicator_with_interface_tag(self):
        @register("indicator", "_IntegTestMACDIndicator", interface="streaming")
        class _MACD:
            pass

        assert GLOBAL_REGISTRY.get_interface("indicator", "_IntegTestMACDIndicator") == "streaming"

    def test_list_names_is_frozenset(self):
        @register("indicator", "_IntegTestBollingerIndicator")
        class _BB:
            pass

        names = GLOBAL_REGISTRY.list_names("indicator")
        assert isinstance(names, frozenset)
        assert "_IntegTestBollingerIndicator" in names


# ---------------------------------------------------------------------------
# 3. Portfolio kind
# ---------------------------------------------------------------------------

class TestPortfolioRegistration:
    def test_portfolio_registers_and_retrieves(self):
        cls, retrieved, tag = _register_and_lookup(
            "portfolio", "_IntegTestEqualWeightPortfolio"
        )
        assert retrieved is cls
        assert "portfolio" in GLOBAL_REGISTRY.list_kinds()
        assert "_IntegTestEqualWeightPortfolio" in GLOBAL_REGISTRY.list_names("portfolio")

    def test_portfolio_get_returns_none_for_unknown_name(self):
        result = GLOBAL_REGISTRY.get("portfolio", "_NonExistentPortfolio_xyz")
        assert result is None

    def test_portfolio_with_interface_tag(self):
        @register("portfolio", "_IntegTestMomentumPortfolio", interface="risk_parity")
        class _MP:
            pass

        assert GLOBAL_REGISTRY.get("portfolio", "_IntegTestMomentumPortfolio") is _MP
        assert GLOBAL_REGISTRY.get_interface("portfolio", "_IntegTestMomentumPortfolio") == "risk_parity"


# ---------------------------------------------------------------------------
# 4. Cross-kind isolation
# ---------------------------------------------------------------------------

class TestCrossKindIsolation:
    def test_same_name_different_kinds_do_not_collide(self):
        @register("strategy", "_IntegTestSharedName")
        class _StratCls:
            pass

        @register("indicator", "_IntegTestSharedName")
        class _IndCls:
            pass

        assert GLOBAL_REGISTRY.get("strategy", "_IntegTestSharedName") is _StratCls
        assert GLOBAL_REGISTRY.get("indicator", "_IntegTestSharedName") is _IndCls
        assert GLOBAL_REGISTRY.get("strategy", "_IntegTestSharedName") is not _IndCls

    def test_unknown_kind_returns_empty(self):
        result = GLOBAL_REGISTRY.get("nonexistent_kind", "anything")
        assert result is None
        assert GLOBAL_REGISTRY.list_names("nonexistent_kind") == frozenset()
        assert GLOBAL_REGISTRY.get_all("nonexistent_kind") == {}

    def test_list_kinds_is_frozenset(self):
        kinds = GLOBAL_REGISTRY.list_kinds()
        assert isinstance(kinds, frozenset)

    def test_decorator_returns_class_unchanged(self):
        @register("strategy", "_IntegTestReturnedCls")
        class _Original:
            class_attr = 42

        assert _Original.class_attr == 42
        assert GLOBAL_REGISTRY.get("strategy", "_IntegTestReturnedCls") is _Original
