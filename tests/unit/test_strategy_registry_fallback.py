from __future__ import annotations

import io
from contextlib import redirect_stderr

from lumina_quant.core.plugin_registry import (
    PublicStrategyRegistry,
    PublicStubStrategy,
    import_private_strategy_registry,
    load_strategy_registry,
)
import lumina_quant.strategies as strategies_package


def test_import_private_strategy_registry_resolves_registry_module():
    registry_module = import_private_strategy_registry()

    assert registry_module.__name__ == "lumina_quant.strategies.registry"
    assert registry_module.DEFAULT_STRATEGY_NAME == strategies_package.DEFAULT_STRATEGY_NAME


def test_load_strategy_registry_uses_public_fallback_when_import_fails():
    def _raise():
        raise RuntimeError("boom")

    captured = io.StringIO()
    with redirect_stderr(captured):
        registry = load_strategy_registry(_raise)

    assert isinstance(registry, PublicStrategyRegistry)
    assert registry.DEFAULT_STRATEGY_NAME == "PublicStubStrategy"
    assert registry.get_strategy_map() == {"PublicStubStrategy": PublicStubStrategy}
    stderr_text = captured.getvalue()
    assert "using public fallback" in stderr_text
    assert "RuntimeError: boom" in stderr_text


def test_strategy_package_exports_registry_attribute():
    assert strategies_package.registry.__name__ == "lumina_quant.strategies.registry"
