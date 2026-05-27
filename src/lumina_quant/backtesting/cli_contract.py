"""Shared raw-first/legacy CLI data-mode contract helpers."""

from __future__ import annotations

from dataclasses import dataclass


class RawFirstDataMissingError(RuntimeError):
    """Raised when raw-first mode cannot resolve committed market data."""


class RawFirstManifestInvalidError(RuntimeError):
    """Raised when committed manifest metadata is invalid."""


class RawFirstStaleWindowError(RuntimeError):
    """Raised when committed window watermark is too stale for raw-first mode."""

    def __init__(
        self,
        message: str,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        lag_ms: int | None = None,
        commit_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.symbol = symbol
        self.timeframe = timeframe
        self.lag_ms = lag_ms
        self.commit_id = commit_id


@dataclass(slots=True, frozen=True)
class ResolvedDataContract:
    """Normalized contract values shared by backtest and optimize entrypoints."""

    data_mode: str
    backtest_mode: str
    data_source: str


def normalize_data_mode(value: str | None, default: str = "raw-first") -> str:
    token = str(value or default).strip().lower()
    if token in {"raw-first", "legacy"}:
        return token
    raise RawFirstDataMissingError(
        f"Unsupported data-mode '{value}'. Expected one of: raw-first, legacy."
    )


def normalize_data_source(value: str | None, default: str = "auto") -> str:
    token = str(value or default).strip().lower()
    if token in {"auto", "db", "csv", "external"}:
        return token
    raise RawFirstDataMissingError(
        f"Unsupported data-source '{value}'. Expected one of: auto, db, csv, external."
    )


def normalize_backtest_mode(value: str | None, default: str = "windowed") -> str:
    token = str(value or default).strip().lower()
    if token in {"windowed", "legacy_batch", "legacy_1s"}:
        return token
    raise RawFirstDataMissingError(
        f"Unsupported backtest-mode '{value}'. Expected one of: windowed, legacy_batch, legacy_1s."
    )


def resolve_data_mode_contract(
    *,
    data_mode: str | None,
    backtest_mode: str | None,
    data_source: str | None,
) -> ResolvedDataContract:
    """Resolve and validate the shared backtest/optimize data-loading contract."""
    resolved_data_mode = normalize_data_mode(data_mode, default="raw-first")
    resolved_backtest_mode = normalize_backtest_mode(backtest_mode, default="windowed")
    resolved_data_source = normalize_data_source(data_source, default="auto")

    if resolved_data_mode == "raw-first":
        if resolved_backtest_mode != "windowed":
            raise RawFirstDataMissingError(
                "Invalid combination: --data-mode raw-first requires --backtest-mode windowed."
            )
        if resolved_data_source == "csv":
            raise RawFirstDataMissingError(
                "Invalid combination: --data-mode raw-first cannot be used with --data-source csv."
            )
        if resolved_data_source == "external":
            raise RawFirstDataMissingError(
                "Invalid combination: --data-mode raw-first cannot be used with --data-source external."
            )

    return ResolvedDataContract(
        data_mode=resolved_data_mode,
        backtest_mode=resolved_backtest_mode,
        data_source=resolved_data_source,
    )


def resolve_data_contract(
    *,
    data_mode: str | None,
    backtest_mode: str | None,
    data_source: str | None,
    default_backtest_mode: str = "windowed",
    default_data_source: str = "auto",
) -> ResolvedDataContract:
    """Backward-compatible alias for existing entrypoint call-sites."""
    return resolve_data_mode_contract(
        data_mode=data_mode,
        backtest_mode=(
            backtest_mode if str(backtest_mode or "").strip() else str(default_backtest_mode)
        ),
        data_source=(data_source if str(data_source or "").strip() else str(default_data_source)),
    )


def map_raw_first_exception_to_exit_code(exc: Exception) -> int:
    """Map typed raw-first contract exceptions to CLI exit codes."""
    if isinstance(exc, RawFirstManifestInvalidError):
        return 3
    if isinstance(exc, RawFirstStaleWindowError):
        return 4
    if isinstance(exc, RawFirstDataMissingError):
        return 2
    return 1


def raw_first_exit_code(exc: Exception) -> int:
    """Backward-compatible alias for legacy helper name."""
    return map_raw_first_exception_to_exit_code(exc)


__all__ = [
    "RawFirstDataMissingError",
    "RawFirstManifestInvalidError",
    "RawFirstStaleWindowError",
    "ResolvedDataContract",
    "map_raw_first_exception_to_exit_code",
    "normalize_backtest_mode",
    "normalize_data_mode",
    "normalize_data_source",
    "raw_first_exit_code",
    "resolve_data_contract",
    "resolve_data_mode_contract",
]
