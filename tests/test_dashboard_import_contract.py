from __future__ import annotations

from pathlib import Path


def test_dashboard_modules_do_not_mutate_sys_path() -> None:
    """Guard: dashboard service modules must not call sys.path.insert at module level."""
    root = Path(__file__).resolve().parents[1]
    targets = [
        # retired_stub.py deleted in Phase 6 — removed from list
        root / "src" / "lumina_quant" / "dashboard" / "exact_window_bundle.py",
        root / "src" / "lumina_quant" / "dashboard" / "state_store_service.py",
        root / "src" / "lumina_quant" / "dashboard" / "bridge.py",
        root / "src" / "lumina_quant" / "dashboard" / "cutover_surfaces_service.py",
    ]

    for path in targets:
        source = path.read_text(encoding="utf-8")
        assert "sys.path.insert" not in source, path.as_posix()
