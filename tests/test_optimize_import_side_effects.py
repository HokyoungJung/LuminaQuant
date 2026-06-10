from __future__ import annotations

import sys


def test_optimize_import_does_not_materialize_runtime_config_snapshots(monkeypatch):
    """Importing cli.optimize must NOT call get_default_runtime_config() at module level.

    Config should only be loaded when main() or _current_optimize_runtime_settings()
    is called, not on import.
    """
    monkeypatch.delenv("LQ__BACKTEST__DECISION_CADENCE_SECONDS", raising=False)

    call_log: list[str] = []

    # Spy: track any eager call to get_default_runtime_config at import time
    original_fn = None
    try:
        import lumina_quant.configuration as _cfg_mod
        original_fn = _cfg_mod.get_default_runtime_config

        def _spy():
            call_log.append("called")
            return original_fn()

        monkeypatch.setattr(_cfg_mod, "get_default_runtime_config", _spy)
        # Also patch in all sub-modules that may have already imported it
        for mod_name in list(sys.modules):
            if mod_name.startswith("lumina_quant.") and mod_name != "lumina_quant.configuration":
                mod = sys.modules[mod_name]
                if getattr(mod, "get_default_runtime_config", None) is original_fn:
                    monkeypatch.setattr(mod, "get_default_runtime_config", _spy)
    except Exception:
        pass

    sys.modules.pop("lumina_quant.cli.optimize", None)
    optimize_module = __import__("lumina_quant.cli.optimize", fromlist=["optimize"])

    assert optimize_module.CSV_DIR == "data"
    # No eager config load at import time
    assert call_log == [], (
        f"get_default_runtime_config() was called {len(call_log)} time(s) during import"
    )
    assert "LQ__BACKTEST__DECISION_CADENCE_SECONDS" not in __import__("os").environ
