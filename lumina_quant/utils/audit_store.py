import json
import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime


class AuditStore:
    """Lightweight SQLite audit store for runs, orders, fills and risk events."""

    def __init__(self, db_path="lumina_quant.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _utcnow(self):
        return datetime.now(UTC).isoformat()

    def _ensure_schema(self):
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    status TEXT,
                    client_order_id TEXT,
                    exchange_order_id TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    fill_time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    fill_price REAL,
                    fill_cost REAL,
                    commission REAL,
                    client_order_id TEXT,
                    exchange_order_id TEXT,
                    status TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timeindex TEXT NOT NULL,
                    total REAL NOT NULL,
                    cash REAL,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    details TEXT
                );

                CREATE TABLE IF NOT EXISTS heartbeats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    heartbeat_time TEXT NOT NULL,
                    status TEXT,
                    details TEXT
                );

                CREATE TABLE IF NOT EXISTS order_state_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    symbol TEXT,
                    client_order_id TEXT,
                    exchange_order_id TEXT,
                    state TEXT NOT NULL,
                    message TEXT,
                    details TEXT
                );
                """
            )
            self._conn.commit()

    def start_run(self, mode, metadata=None):
        run_id = str(uuid.uuid4())
        now = self._utcnow()
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs(run_id, mode, started_at, status, metadata) VALUES (?, ?, ?, ?, ?)",
                (run_id, mode, now, "RUNNING", json.dumps(metadata or {})),
            )
            self._conn.commit()
        return run_id

    def end_run(self, run_id, status="COMPLETED", metadata=None):
        now = self._utcnow()
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET ended_at=?, status=?, metadata=? WHERE run_id=?",
                (now, status, json.dumps(metadata or {}), run_id),
            )
            self._conn.commit()

    def log_order(self, run_id, order_event, status="NEW", exchange_order_id=None):
        payload = order_event.metadata if hasattr(order_event, "metadata") else {}
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO orders(run_id, created_at, symbol, side, order_type, quantity, price, status,
                                   client_order_id, exchange_order_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    self._utcnow(),
                    order_event.symbol,
                    order_event.direction,
                    order_event.order_type,
                    float(order_event.quantity),
                    float(order_event.price) if order_event.price is not None else None,
                    status,
                    order_event.client_order_id,
                    exchange_order_id,
                    json.dumps(payload or {}),
                ),
            )
            self._conn.commit()

    def log_fill(self, run_id, fill_event):
        fill_price = None
        if fill_event.fill_cost is not None and fill_event.quantity:
            fill_price = float(fill_event.fill_cost) / float(fill_event.quantity)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO fills(run_id, fill_time, symbol, side, quantity, fill_price, fill_cost, commission,
                                  client_order_id, exchange_order_id, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(fill_event.timeindex),
                    fill_event.symbol,
                    fill_event.direction,
                    float(fill_event.quantity),
                    fill_price,
                    float(fill_event.fill_cost) if fill_event.fill_cost is not None else None,
                    float(fill_event.commission) if fill_event.commission is not None else 0.0,
                    fill_event.client_order_id,
                    fill_event.order_id,
                    fill_event.status,
                    json.dumps(fill_event.metadata or {}),
                ),
            )
            self._conn.commit()

    def log_equity(self, run_id, timeindex, total, cash=None, metadata=None):
        with self._lock:
            self._conn.execute(
                "INSERT INTO equity(run_id, timeindex, total, cash, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    str(timeindex),
                    float(total),
                    float(cash) if cash is not None else None,
                    json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()

    def log_risk_event(self, run_id, reason, details=None):
        with self._lock:
            self._conn.execute(
                "INSERT INTO risk_events(run_id, event_time, reason, details) VALUES (?, ?, ?, ?)",
                (run_id, self._utcnow(), reason, json.dumps(details or {})),
            )
            self._conn.commit()

    def log_heartbeat(self, run_id, status="ALIVE", details=None):
        with self._lock:
            self._conn.execute(
                "INSERT INTO heartbeats(run_id, heartbeat_time, status, details) VALUES (?, ?, ?, ?)",
                (
                    run_id,
                    self._utcnow(),
                    status,
                    json.dumps(details or {}),
                ),
            )
            self._conn.commit()

    def log_order_state(self, run_id, state_payload):
        details = dict(state_payload.get("metadata") or {})
        if "last_filled" in state_payload:
            details["last_filled"] = state_payload["last_filled"]
        if "created_at" in state_payload:
            details["created_at"] = state_payload["created_at"]

        symbol = state_payload.get("symbol")
        client_order_id = state_payload.get("client_order_id")
        exchange_order_id = state_payload.get("order_id")
        state = str(state_payload.get("state", "UNKNOWN"))
        message = state_payload.get("message")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO order_state_events(
                    run_id, event_time, symbol, client_order_id, exchange_order_id, state, message, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    self._utcnow(),
                    symbol,
                    client_order_id,
                    exchange_order_id,
                    state,
                    message,
                    json.dumps(details),
                ),
            )
            if client_order_id:
                self._conn.execute(
                    """
                    UPDATE orders
                    SET status = ?, exchange_order_id = COALESCE(?, exchange_order_id)
                    WHERE run_id = ? AND client_order_id = ?
                    """,
                    (state, exchange_order_id, run_id, client_order_id),
                )
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()
