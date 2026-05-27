from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "lumina_quant" / "dashboard" / "state_store_service.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dashboard_state_store_service", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load dashboard state-store service")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_postgres_dsn_prefers_explicit_value(monkeypatch) -> None:
    module = _load_module()

    class _Config:
        POSTGRES_DSN = "postgres://config"

    monkeypatch.setenv("LQ_POSTGRES_DSN", "postgres://env")

    assert (
        module.resolve_postgres_dsn("postgres://explicit", base_config=_Config)
        == "postgres://explicit"
    )


def test_execute_query_logs_when_fetchall_fails(caplog) -> None:
    module = _load_module()
    caplog.set_level("WARNING")

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query: str, params: tuple[Any, ...]) -> None:
            self.query = query
            self.params = params

        def fetchall(self):
            raise RuntimeError("fetchall failed")

        def close(self) -> None:
            return None

    class _Conn:
        def __init__(self):
            self.cursor_obj = _Cursor()
            self.committed = False
            self.closed = False

        def cursor(self):
            return module.StateCursor(self.cursor_obj)

        def commit(self):
            self.committed = True

        def close(self):
            self.closed = True

    conn = _Conn()

    rows = module.execute_query(
        "postgres://lumina",
        "SELECT 1",
        connect_state_store=lambda _dsn: conn,
    )

    assert rows == []
    assert conn.committed is True
    assert conn.closed is True
    assert any("fell back to an empty result set" in record.message for record in caplog.records)
