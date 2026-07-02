"""Snapshot-first, source-pinned TradFi research fetcher (gated, replay-only).

Research-only external adapter for a small set of free, no-key TradFi endpoints
(Yahoo Finance download CSV, Stooq daily CSV, SEC EDGAR submissions JSON).  The
design is deliberately conservative:

* **Env-gated network I/O.**  Any live HTTP call requires BOTH the runtime config
  flag (``ResearchConfig.tradfi_external_fetch.enabled`` /
  ``allow_network``) AND the ``LUMINA_ENABLE_TRADFI_EXTERNAL_FETCH`` environment
  gate (default OFF).  With the gate OFF the fetcher is replay-only: it reads a
  previously captured on-disk snapshot and never touches the network.
* **Source-pinned, fail-loud.**  A caller picks exactly one provider.  There is
  NO automatic fallback chain — if the pinned provider's snapshot is missing and
  network is not authorized, a :class:`TradFiExternalFetchError` is raised loudly
  instead of silently trying another source.
* **Free-unauthenticated boundary.**  Every provider source record must pass
  ``assert_source_registry_is_free_unauthenticated`` at import/registry build
  time, so a paid / key-required source can never be wired in here.

No default runtime path imports this module; crypto/perp live and golden numerics
are untouched.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumina_quant.research.external_source_registry import (
    ExternalSourceRecord,
    assert_source_registry_is_free_unauthenticated,
)

TRADFI_EXTERNAL_FETCH_ENV = "LUMINA_ENABLE_TRADFI_EXTERNAL_FETCH"

VALID_PROVIDERS = frozenset({"yahoo", "stooq", "sec_edgar"})


class TradFiExternalFetchError(RuntimeError):
    """Raised when a source-pinned TradFi fetch cannot be satisfied (fail-loud)."""


@dataclass(frozen=True, slots=True)
class TradFiFetchResult:
    """Result of a replay/fetch: provider provenance + observation records."""

    provider: str
    symbol: str
    source_id: str
    from_snapshot: bool
    snapshot_path: str
    observations: tuple[dict[str, Any], ...]


def _provider_source_records() -> dict[str, ExternalSourceRecord]:
    records = {
        "yahoo": ExternalSourceRecord(
            source_id="yahoo_finance_download_csv_no_key",
            url="https://query1.finance.yahoo.com/v7/finance/download/{symbol}",
            license_note="Free no-key historical CSV download; verify terms before redistribution.",
            credential_requirement="none",
            update_cadence="daily public bar refresh",
            release_lag_policy="feature availability must lag the public bar timestamp",
            cache_path="var/cache/tradfi_external/yahoo/{symbol}.json",
            allowed_usage_label="research_only",
        ),
        "stooq": ExternalSourceRecord(
            source_id="stooq_daily_csv_no_key",
            url="https://stooq.com/q/d/l/?s={symbol}&i=d",
            license_note="Free no-key daily CSV endpoint; verify terms before redistribution.",
            credential_requirement="none",
            update_cadence="daily public bar refresh",
            release_lag_policy="feature availability must lag the public bar timestamp",
            cache_path="var/cache/tradfi_external/stooq/{symbol}.json",
            allowed_usage_label="research_only",
        ),
        "sec_edgar": ExternalSourceRecord(
            source_id="sec_edgar_submissions_no_key",
            url="https://data.sec.gov/submissions/CIK{symbol}.json",
            license_note="Free public SEC EDGAR submissions JSON; respect fair-access rate limits.",
            credential_requirement="none",
            update_cadence="filing-driven public refresh",
            release_lag_policy="feature availability must lag the filing acceptance timestamp",
            cache_path="var/cache/tradfi_external/sec_edgar/{symbol}.json",
            allowed_usage_label="research_only",
        ),
    }
    # Fail loud at build time if any pinned source is not free/unauthenticated.
    assert_source_registry_is_free_unauthenticated(records.values())
    return records


def tradfi_source_record(provider: str) -> ExternalSourceRecord:
    """Return the pinned free/no-key source record for ``provider`` (fail-loud)."""
    key = str(provider or "").strip().lower()
    records = _provider_source_records()
    try:
        return records[key]
    except KeyError as exc:
        raise TradFiExternalFetchError(
            f"unknown TradFi provider {provider!r}; valid: {sorted(VALID_PROVIDERS)}"
        ) from exc


def _env_truthy(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _compact_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace("/", "_").replace("\\", "_")


def _snapshot_path(provider: str, symbol: str, snapshot_dir: str | Path) -> Path:
    root = Path(snapshot_dir).expanduser()
    return root / provider / f"{_compact_symbol(symbol)}.json"


def write_snapshot(
    *,
    provider: str,
    symbol: str,
    observations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    snapshot_dir: str | Path,
) -> Path:
    """Persist a deterministic replay snapshot for a provider/symbol pair."""
    key = str(provider or "").strip().lower()
    record = tradfi_source_record(key)
    path = _snapshot_path(key, symbol, snapshot_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": key,
        "symbol": str(symbol),
        "source_id": record.source_id,
        "observations": [dict(item) for item in observations],
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return path


def _read_snapshot(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TradFiExternalFetchError(f"malformed TradFi snapshot at {path}")
    return raw


def _fetch_over_network(
    *,
    record: ExternalSourceRecord,
    symbol: str,
) -> list[dict[str, Any]]:  # pragma: no cover - only reached behind both gates
    """Perform the actual authorized HTTP fetch (behind both gates).

    Intentionally minimal and isolated so the default (gated-OFF) path never
    imports networking libraries.  Kept out of unit coverage — the fetcher is
    exercised through the deterministic replay path.
    """
    import urllib.request

    url = record.url.replace("{symbol}", _compact_symbol(symbol))
    request = urllib.request.Request(url, headers={"User-Agent": "lumina-quant-research/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
    return [{"raw": body}]


def fetch_tradfi_series(
    *,
    provider: str,
    symbol: str,
    snapshot_dir: str | Path,
    allow_network: bool = False,
    env: Mapping[str, str] | None = None,
) -> TradFiFetchResult:
    """Replay a snapshot for ``(provider, symbol)``, or fetch it iff authorized.

    Resolution order (source-pinned, no fallback):

    1.  If a snapshot exists on disk -> replay it (always allowed, no network).
    2.  Else, network fetch is attempted ONLY when all authorizations hold:
        ``allow_network`` is True AND the ``LUMINA_ENABLE_TRADFI_EXTERNAL_FETCH``
        environment gate is truthy.  On success the snapshot is written for
        deterministic replay.
    3.  Otherwise raise :class:`TradFiExternalFetchError` (fail-loud).
    """
    effective_env = os.environ if env is None else env
    key = str(provider or "").strip().lower()
    record = tradfi_source_record(key)
    path = _snapshot_path(key, symbol, snapshot_dir)

    if path.exists():
        payload = _read_snapshot(path)
        observations = tuple(
            dict(item) for item in payload.get("observations", []) if isinstance(item, dict)
        )
        return TradFiFetchResult(
            provider=key,
            symbol=str(symbol),
            source_id=str(payload.get("source_id") or record.source_id),
            from_snapshot=True,
            snapshot_path=str(path),
            observations=observations,
        )

    env_gate = _env_truthy(effective_env, TRADFI_EXTERNAL_FETCH_ENV)
    if not (allow_network and env_gate):
        raise TradFiExternalFetchError(
            f"no snapshot for provider={key!r} symbol={symbol!r} at {path} and network "
            f"fetch is not authorized (allow_network={allow_network}, "
            f"{TRADFI_EXTERNAL_FETCH_ENV}={'set' if env_gate else 'unset'}); "
            "source-pinned fetcher does not fall back to another provider."
        )

    observations = tuple(_fetch_over_network(record=record, symbol=symbol))
    write_snapshot(
        provider=key,
        symbol=symbol,
        observations=list(observations),
        snapshot_dir=snapshot_dir,
    )
    return TradFiFetchResult(
        provider=key,
        symbol=str(symbol),
        source_id=record.source_id,
        from_snapshot=False,
        snapshot_path=str(path),
        observations=observations,
    )


__all__ = [
    "TRADFI_EXTERNAL_FETCH_ENV",
    "VALID_PROVIDERS",
    "TradFiExternalFetchError",
    "TradFiFetchResult",
    "fetch_tradfi_series",
    "tradfi_source_record",
    "write_snapshot",
]
