"""Optimization storage helpers."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from typing import Any


def save_optimization_rows(
    db_path: str, run_id: str, stage: str, rows: list[dict[str, Any]]
) -> None:
    """Persist optimization rows into SQLite."""
    if not rows:
        return

    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            created_at TEXT NOT NULL,
            params_json TEXT NOT NULL,
            sharpe REAL,
            cagr TEXT,
            mdd TEXT,
            train_sharpe REAL,
            robustness_score REAL,
            extra_json TEXT
        )
        """
    )

    now = datetime.now(UTC).isoformat()
    for row in rows:
        cur.execute(
            """
            INSERT INTO optimization_results(
                run_id, stage, created_at, params_json, sharpe, cagr, mdd, train_sharpe, robustness_score, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                stage,
                now,
                json.dumps(row.get("params", {})),
                float(row.get("sharpe", 0.0)) if row.get("sharpe") is not None else None,
                str(row.get("cagr")) if row.get("cagr") is not None else None,
                str(row.get("mdd")) if row.get("mdd") is not None else None,
                float(row.get("train_sharpe", 0.0))
                if row.get("train_sharpe") is not None
                else None,
                float(row.get("robustness_score", 0.0))
                if row.get("robustness_score") is not None
                else None,
                json.dumps(
                    {
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "params",
                            "sharpe",
                            "cagr",
                            "mdd",
                            "train_sharpe",
                            "robustness_score",
                        }
                    }
                ),
            ),
        )
    conn.commit()
    conn.close()
