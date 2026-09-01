"""Shared workflow helpers."""

from __future__ import annotations

import os
import uuid


def resolve_run_id(explicit_run_id: str | None = None) -> str:
    token = str(explicit_run_id or "").strip()
    return token or str(uuid.uuid4())


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}
