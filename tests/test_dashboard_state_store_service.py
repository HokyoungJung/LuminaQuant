from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


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


def _coerce_datetime(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    frame = frame.copy()
    frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return frame


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def test_load_metrics_state_frame_reads_funding_metadata_key(monkeypatch) -> None:
    """Equity metadata rows carrying 'funding' must yield a non-zero column.

    Writers emit 'funding' (a cumulative quote-currency amount, see
    cli/backtest.py); the reader used to key off 'funding_total' — a key no
    writer emits — flattening every funding curve to zero. 'funding_total'
    stays supported as a legacy fallback only.
    """
    module = _load_module()
    frame = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "total": 1000.0,
                "cash": 900.0,
                "metadata": '{"benchmark_price": 100.0, "funding": -3.5, "symbol": "BTC/USDT"}',
            },
            {
                "datetime": "2026-03-21T01:00:00Z",
                "total": 1001.0,
                "cash": 901.0,
                "metadata": '{"benchmark_price": 101.0, "funding_total": -4.25}',
            },
            {
                "datetime": "2026-03-21T02:00:00Z",
                "total": 1002.0,
                "cash": 902.0,
                "metadata": "{}",
            },
        ]
    )
    monkeypatch.setattr(module.pd, "read_sql_query", lambda query, conn, params=None: frame.copy())

    class _Conn:
        def close(self) -> None:
            return None

    result = module.load_metrics_state_frame(
        "postgres://x",
        "run-123",
        connect_state_store=lambda dsn: _Conn(),
        coerce_datetime=_coerce_datetime,
        parse_json_dict=_parse_json_dict,
        max_points=10,
    )

    assert result["funding"].tolist() == [-3.5, -4.25, 0.0]
    assert result["benchmark_price"].iloc[0] == 100.0
    assert result["event_symbol"].iloc[0] == "BTC/USDT"
