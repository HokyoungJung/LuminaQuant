"""Gate-OFF / replay / free-unauthenticated invariants for the TradFi fetcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.research import tradfi_fetcher as tf
from lumina_quant.research.external_source_registry import (
    validate_source_registry,
)


def test_config_defaults_are_off() -> None:
    cfg = get_default_runtime_config().research.tradfi_external_fetch
    assert cfg.enabled is False
    assert cfg.allow_network is False
    assert cfg.provider == "yahoo"


@pytest.mark.parametrize("provider", sorted(tf.VALID_PROVIDERS))
def test_all_provider_sources_are_free_unauthenticated(provider: str) -> None:
    record = tf.tradfi_source_record(provider)
    (decision,) = validate_source_registry([record])
    assert decision.allowed, decision.reasons


def test_missing_snapshot_without_authorization_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(tf.TradFiExternalFetchError):
        tf.fetch_tradfi_series(
            provider="yahoo",
            symbol="AAPL",
            snapshot_dir=tmp_path,
            allow_network=False,
            env={},
        )


def test_env_gate_default_off_blocks_even_with_allow_network(tmp_path: Path) -> None:
    # allow_network True but env gate unset -> still blocked (source-pinned, no I/O).
    with pytest.raises(tf.TradFiExternalFetchError):
        tf.fetch_tradfi_series(
            provider="stooq",
            symbol="SPY",
            snapshot_dir=tmp_path,
            allow_network=True,
            env={},
        )


def test_replay_from_snapshot_needs_no_network(tmp_path: Path) -> None:
    tf.write_snapshot(
        provider="yahoo",
        symbol="AAPL",
        observations=[{"date": "2026-01-02", "close": 190.5}],
        snapshot_dir=tmp_path,
    )
    result = tf.fetch_tradfi_series(
        provider="yahoo",
        symbol="AAPL",
        snapshot_dir=tmp_path,
        allow_network=False,  # replay never needs authorization
        env={},
    )
    assert result.from_snapshot is True
    assert result.source_id == "yahoo_finance_download_csv_no_key"
    assert result.observations == ({"date": "2026-01-02", "close": 190.5},)


def test_unknown_provider_fails_loud() -> None:
    with pytest.raises(tf.TradFiExternalFetchError):
        tf.tradfi_source_record("bloomberg_terminal")


def test_no_automatic_fallback_between_providers(tmp_path: Path) -> None:
    # A snapshot for one provider must not satisfy a pinned request for another.
    tf.write_snapshot(
        provider="stooq",
        symbol="SPY",
        observations=[{"date": "2026-01-02", "close": 470.0}],
        snapshot_dir=tmp_path,
    )
    with pytest.raises(tf.TradFiExternalFetchError):
        tf.fetch_tradfi_series(
            provider="yahoo",
            symbol="SPY",
            snapshot_dir=tmp_path,
            allow_network=False,
            env={tf.TRADFI_EXTERNAL_FETCH_ENV: "true"},
        )
