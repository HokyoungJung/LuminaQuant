from __future__ import annotations

import pytest

from lumina_quant.research.external_source_registry import (
    ExternalSourceRecord,
    SourceAccess,
    assert_source_registry_is_free_unauthenticated,
    build_tradfi_external_alpha_source_registry,
    validate_external_source_boundary,
    validate_source_registry,
)


def _source(**overrides) -> ExternalSourceRecord:
    payload = {
        "source_id": "public-csv",
        "url": "https://example.com/public.csv",
        "license_note": "Free public data; verify terms before redistribution.",
        "credential_requirement": "none",
        "update_cadence": "daily",
        "release_lag_policy": "lag by public release timestamp",
        "cache_path": "var/cache/external_sources/public.csv",
        "allowed_usage_label": "diagnostic_only",
    }
    payload.update(overrides)
    return ExternalSourceRecord(**payload)


def test_rejects_paid_api_key_broker_and_live_sources() -> None:
    paid = _source(
        source_id="paid-vendor",
        access=SourceAccess.PAID,
        credential_requirement="paid_subscription",
        is_paid=True,
    )
    api_keyed = _source(
        source_id="api-keyed",
        access=SourceAccess.FREE_API_KEY,
        credential_requirement="api_key",
        requires_api_key=True,
    )
    broker = _source(
        source_id="broker-feed",
        access=SourceAccess.BROKER_CREDENTIAL,
        credential_requirement="broker_credentials",
        requires_broker_credentials=True,
    )
    live_account = _source(
        source_id="live-account-feed",
        access=SourceAccess.LIVE_AUTHENTICATED,
        credential_requirement="account_login",
        requires_live_account=True,
    )

    decisions = validate_source_registry([paid, api_keyed, broker, live_account])

    assert {decision.source_id for decision in decisions if not decision.allowed} == {
        "paid-vendor",
        "api-keyed",
        "broker-feed",
        "live-account-feed",
    }
    assert any("paid_source" in decision.reasons for decision in decisions)
    assert any("requires_api_key" in decision.reasons for decision in decisions)
    assert any("requires_broker_credentials" in decision.reasons for decision in decisions)
    assert any("requires_live_account" in decision.reasons for decision in decisions)
    with pytest.raises(ValueError, match="paid-vendor"):
        assert_source_registry_is_free_unauthenticated([paid, api_keyed, broker, live_account])


def test_missing_or_unknown_metadata_fails_closed() -> None:
    decision = validate_external_source_boundary(
        {
            "source_id": "mystery",
            "url": "",
            "license_note": "",
            "credential_requirement": "registration",
            "update_cadence": "",
            "release_lag_policy": "",
            "cache_path": "",
            "allowed_usage_label": "",
            "access": "unknown",
        }
    )

    assert decision.allowed is False
    assert "missing_url" in decision.reasons
    assert "missing_release_lag_policy" in decision.reasons
    assert "access_not_free_unauthenticated:unknown" in decision.reasons
    assert "credential_requirement_not_none:registration" in decision.reasons


def test_allows_only_free_unauthenticated_sources_with_required_metadata() -> None:
    record = _source(credential_requirement="unauthenticated")

    decision = validate_external_source_boundary(record)

    assert decision.allowed is True
    assert decision.reasons == ()
    assert_source_registry_is_free_unauthenticated([record])


def test_tradfi_external_alpha_registry_is_no_key_and_metadata_complete() -> None:
    records = build_tradfi_external_alpha_source_registry()

    decisions = assert_source_registry_is_free_unauthenticated(records)

    assert len(records) >= 5
    assert all(decision.allowed for decision in decisions)
    payloads = [record.to_dict() for record in records]
    assert all(payload["access"] == "free_unauthenticated" for payload in payloads)
    assert all(payload["credential_requirement"] == "none" for payload in payloads)
    assert all(payload["cache_path"] for payload in payloads)
    assert all(payload["release_lag_policy"] for payload in payloads)
    assert {payload["source_id"] for payload in payloads} >= {
        "fama_french_daily_factors",
        "fred_graph_csv_no_key",
        "nyse_hours_calendars",
    }
