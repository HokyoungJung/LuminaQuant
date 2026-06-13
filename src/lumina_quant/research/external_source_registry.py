"""External evidence/data source boundary checks for research manifests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class SourceAccess(StrEnum):
    FREE_UNAUTHENTICATED = "free_unauthenticated"
    FREE_API_KEY = "free_api_key"
    AUTHENTICATED = "authenticated"
    LIVE_AUTHENTICATED = "live_authenticated"
    BROKER_CREDENTIAL = "broker_credential"
    PAID = "paid"
    UNKNOWN = "unknown"


ALLOWED_CREDENTIAL_REQUIREMENTS = frozenset(
    {
        "none",
        "no",
        "no_key",
        "not_required",
        "unauthenticated",
        "free_unauthenticated",
    }
)

ALLOWED_USAGE_LABELS = frozenset(
    {
        "evidence_only",
        "strategy_class_evidence",
        "diagnostic_only",
        "research_only",
        "clean_candidate",
        "shadow_freeze_only",
    }
)


@dataclass(frozen=True, slots=True)
class ExternalSourceRecord:
    source_id: str
    url: str
    license_note: str
    credential_requirement: str
    update_cadence: str
    release_lag_policy: str
    cache_path: str
    allowed_usage_label: str
    fallback_behavior: str = "fail_closed"
    access: SourceAccess = SourceAccess.FREE_UNAUTHENTICATED
    is_paid: bool = False
    requires_api_key: bool = False
    requires_broker_credentials: bool = False
    requires_live_account: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ExternalSourceRecord:
        access = _normalize_access(
            payload.get("access")
            or payload.get("access_model")
            or payload.get("source_access")
            or SourceAccess.FREE_UNAUTHENTICATED
        )
        notes_value = payload.get("notes") or ()
        if isinstance(notes_value, str):
            notes = (notes_value,)
        else:
            notes = tuple(str(item) for item in notes_value)
        return cls(
            source_id=str(payload.get("source_id") or payload.get("id") or ""),
            url=str(payload.get("url") or payload.get("source_url") or ""),
            license_note=str(payload.get("license_note") or payload.get("license") or ""),
            credential_requirement=str(payload.get("credential_requirement") or ""),
            update_cadence=str(payload.get("update_cadence") or ""),
            release_lag_policy=str(
                payload.get("release_lag_policy") or payload.get("release_lag") or ""
            ),
            cache_path=str(payload.get("cache_path") or payload.get("cache_policy") or ""),
            allowed_usage_label=str(payload.get("allowed_usage_label") or ""),
            fallback_behavior=str(payload.get("fallback_behavior") or ""),
            access=access,
            is_paid=bool(payload.get("is_paid", False)),
            requires_api_key=bool(payload.get("requires_api_key", False)),
            requires_broker_credentials=bool(payload.get("requires_broker_credentials", False)),
            requires_live_account=bool(payload.get("requires_live_account", False)),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["access"] = self.access.value
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True, slots=True)
class SourceBoundaryDecision:
    source_id: str
    allowed: bool
    reasons: tuple[str, ...]


def _normalize_access(value: object) -> SourceAccess:
    if isinstance(value, SourceAccess):
        return value
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return SourceAccess(token)
    except ValueError:
        return SourceAccess.UNKNOWN


def _is_blank(value: object) -> bool:
    return not str(value or "").strip()


def _credential_token(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _coerce_record(record: ExternalSourceRecord | Mapping[str, Any]) -> ExternalSourceRecord:
    if isinstance(record, ExternalSourceRecord):
        return record
    return ExternalSourceRecord.from_mapping(record)


def validate_external_source_boundary(
    record: ExternalSourceRecord | Mapping[str, Any],
) -> SourceBoundaryDecision:
    source = _coerce_record(record)
    reasons: list[str] = []

    required_fields = {
        "source_id": source.source_id,
        "url": source.url,
        "license_note": source.license_note,
        "update_cadence": source.update_cadence,
        "release_lag_policy": source.release_lag_policy,
        "cache_path": source.cache_path,
        "fallback_behavior": source.fallback_behavior,
    }
    for field_name, value in required_fields.items():
        if _is_blank(value):
            reasons.append(f"missing_{field_name}")

    access = _normalize_access(source.access)
    if access is not SourceAccess.FREE_UNAUTHENTICATED:
        reasons.append(f"access_not_free_unauthenticated:{access.value}")

    credential_requirement = _credential_token(source.credential_requirement)
    if credential_requirement not in ALLOWED_CREDENTIAL_REQUIREMENTS:
        reasons.append(f"credential_requirement_not_none:{credential_requirement or 'missing'}")

    usage_label = _credential_token(source.allowed_usage_label)
    if usage_label not in ALLOWED_USAGE_LABELS:
        reasons.append(f"unsupported_allowed_usage_label:{usage_label or 'missing'}")

    if source.is_paid:
        reasons.append("paid_source")
    if source.requires_api_key:
        reasons.append("requires_api_key")
    if source.requires_broker_credentials:
        reasons.append("requires_broker_credentials")
    if source.requires_live_account:
        reasons.append("requires_live_account")

    return SourceBoundaryDecision(
        source_id=source.source_id or "<missing>",
        allowed=not reasons,
        reasons=tuple(reasons),
    )


def validate_source_registry(
    records: Iterable[ExternalSourceRecord | Mapping[str, Any]],
) -> tuple[SourceBoundaryDecision, ...]:
    return tuple(validate_external_source_boundary(record) for record in records)


def assert_source_registry_is_free_unauthenticated(
    records: Iterable[ExternalSourceRecord | Mapping[str, Any]],
) -> tuple[SourceBoundaryDecision, ...]:
    decisions = validate_source_registry(records)
    rejected = [decision for decision in decisions if not decision.allowed]
    if rejected:
        detail = "; ".join(
            f"{decision.source_id}: {', '.join(decision.reasons)}" for decision in rejected
        )
        raise ValueError(f"external source registry contains disallowed sources: {detail}")
    return decisions


def build_tradfi_external_alpha_source_registry() -> tuple[ExternalSourceRecord, ...]:
    """Return the approved no-key source registry for TradFi external-alpha research."""
    return (
        ExternalSourceRecord(
            source_id="volatility_managed_sizing_ssrn",
            url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2659431",
            license_note="Free public abstract/reference; verify terms before redistribution.",
            credential_requirement="none",
            update_cadence="static literature reference",
            release_lag_policy="strategy-class evidence only; no time-series data consumed",
            cache_path="var/cache/external_sources/volatility_managed_sizing_ssrn.html",
            allowed_usage_label="strategy_class_evidence",
        ),
        ExternalSourceRecord(
            source_id="time_series_momentum_ssrn",
            url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463",
            license_note="Free public abstract/reference; verify terms before redistribution.",
            credential_requirement="none",
            update_cadence="static literature reference",
            release_lag_policy="strategy-class evidence only; no time-series data consumed",
            cache_path="var/cache/external_sources/time_series_momentum_ssrn.html",
            allowed_usage_label="strategy_class_evidence",
        ),
        ExternalSourceRecord(
            source_id="intraday_momentum_ssrn",
            url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2552752",
            license_note="Free public abstract/reference; verify terms before redistribution.",
            credential_requirement="none",
            update_cadence="static literature reference",
            release_lag_policy="strategy-class evidence only; no time-series data consumed",
            cache_path="var/cache/external_sources/intraday_momentum_ssrn.html",
            allowed_usage_label="strategy_class_evidence",
        ),
        ExternalSourceRecord(
            source_id="nyse_hours_calendars",
            url="https://www.nyse.com/trade/hours-calendars",
            license_note="Free public exchange-hours reference; cache only derived metadata.",
            credential_requirement="none",
            update_cadence="exchange calendar updates",
            release_lag_policy="session calendar metadata must be versioned before replay",
            cache_path="var/cache/external_sources/nyse_hours_calendars.json",
            allowed_usage_label="diagnostic_only",
        ),
        ExternalSourceRecord(
            source_id="fama_french_daily_factors",
            url="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
            license_note="Free public academic factor library; verify terms before redistribution.",
            credential_requirement="none",
            update_cadence="periodic data-library refresh",
            release_lag_policy="feature availability must lag the file publication timestamp",
            cache_path="var/cache/external_sources/fama_french_daily_factors.csv",
            allowed_usage_label="diagnostic_only",
        ),
        ExternalSourceRecord(
            source_id="fred_graph_csv_no_key",
            url="https://fredhelp.stlouisfed.org/fred/data/downloading/using-the-download-data-link/",
            license_note="Free no-key graph CSV download path; REST API-key endpoint is excluded.",
            credential_requirement="none",
            update_cadence="series-specific public CSV refresh",
            release_lag_policy="feature availability must lag the public observation/release timestamp",
            cache_path="var/cache/external_sources/fred_graph_csv_no_key/{series_id}.csv",
            allowed_usage_label="diagnostic_only",
            notes=("FRED API-key routes are intentionally not allowed for this cycle.",),
        ),
    )


__all__ = [
    "ExternalSourceRecord",
    "SourceAccess",
    "SourceBoundaryDecision",
    "assert_source_registry_is_free_unauthenticated",
    "build_tradfi_external_alpha_source_registry",
    "validate_external_source_boundary",
    "validate_source_registry",
]
