"""Artifact-only bridge for DeepLearning forecast outputs.

This module deliberately performs no model training or inference. It only reads
forecast artifacts already produced by the external DeepLearning repository and
normalizes them into a small in-memory lookup usable by LuminaQuant strategies.
"""

from __future__ import annotations

import csv
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np

try:  # pragma: no cover - exercised only when parquet artifacts are supplied.
    import polars as pl
except Exception:  # pragma: no cover
    pl = None  # type: ignore[assignment]

SUPPORTED_DEEP_LEARNING_MODELS: Final[tuple[str, ...]] = (
    "FITS",
    "CycleNet",
    "CMamba",
    "PatchTST",
)
_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    "predicted_value",
    "pred_value",
    "prediction",
    "value",
    "Data",
    "data",
)
_RETURN_COLUMNS: Final[tuple[str, ...]] = (
    "pred_return",
    "predicted_return",
    "return",
    "expected_return",
    "forecast_return",
)
_RETURN_BPS_COLUMNS: Final[tuple[str, ...]] = (
    "pred_return_bps",
    "return_bps",
    "expected_return_bps",
    "forecast_return_bps",
)
_CONFIDENCE_COLUMNS: Final[tuple[str, ...]] = (
    "confidence",
    "forecast_confidence",
    "model_confidence",
)
_ORIGIN_TIME_COLUMNS: Final[tuple[str, ...]] = (
    "origin_time",
    "forecast_origin",
    "pred_date",
    "prediction_time",
    "timestamp",
    "datetime",
    "time",
)
_TARGET_TIME_COLUMNS: Final[tuple[str, ...]] = (
    "target_time",
    "forecast_time",
    "Date",
    "date",
    "target_date",
)
_SYMBOL_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "asset",
    "ticker",
    "dbcode",
    "db_code",
    "target",
)
_MODEL_COLUMNS: Final[tuple[str, ...]] = ("model", "model_name", "model_code")
_PRICE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {"open", "high", "low", "close", "volume", "data"}
)


def normalize_deep_learning_models(
    value: Any,
    *,
    default: tuple[str, ...] = SUPPORTED_DEEP_LEARNING_MODELS,
) -> tuple[str, ...]:
    """Normalize user/model-config input to supported DeepLearning model names."""
    if value is None:
        return default
    if isinstance(value, str):
        raw = [part.strip() for part in value.replace(";", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(part).strip() for part in value]
    else:
        raw = [str(value).strip()]

    supported_by_lower = {model.lower(): model for model in SUPPORTED_DEEP_LEARNING_MODELS}
    out: list[str] = []
    for item in raw:
        model = supported_by_lower.get(item.lower())
        if model is not None and model not in out:
            out.append(model)
    if out:
        return tuple(out)
    return default


@dataclass(frozen=True, slots=True)
class DeepLearningForecastRecord:
    symbol: str
    model: str
    origin_time: datetime
    target_time: datetime | None = None
    predicted_value: float | None = None
    predicted_return: float | None = None
    confidence: float | None = None
    source_path: str = ""


@dataclass(frozen=True, slots=True)
class DeepLearningForecastSnapshot:
    symbol: str
    event_time: datetime
    origin_time: datetime
    target_time: datetime | None
    model_returns: dict[str, float]
    model_values: dict[str, float]
    model_confidences: dict[str, float]
    mean_return: float
    dispersion: float
    long_vote_fraction: float
    short_vote_fraction: float
    source_confidence: float

    @property
    def model_count(self) -> int:
        return len(self.model_returns)


def _first_value(row: dict[str, Any], columns: tuple[str, ...]) -> Any:
    for column in columns:
        if column in row and row[column] not in (None, ""):
            return row[column]
    lower_map = {str(key).lower(): key for key in row}
    for column in columns:
        key = lower_map.get(column.lower())
        if key is not None and row[key] not in (None, ""):
            return row[key]
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, np.datetime64):
        try:
            return datetime.fromtimestamp(
                value.astype("datetime64[ms]").astype("int64") / 1000.0,
                tz=UTC,
            )
        except Exception:
            return None
    if isinstance(value, (int, float)):
        raw = float(value)
        if not math.isfinite(raw):
            return None
        if abs(raw) > 1_000_000_000_000_000:
            raw /= 1_000_000_000.0
        elif abs(raw) > 1_000_000_000_000:
            raw /= 1000.0
        try:
            return datetime.fromtimestamp(raw, tz=UTC)
        except Exception:
            return None
    token = str(value).strip()
    if not token:
        return None
    try:
        numeric = float(token)
    except Exception:
        numeric = None
    if numeric is not None and math.isfinite(numeric):
        return _parse_timestamp(numeric)
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except Exception:
        try:
            parsed = datetime.strptime(token, "%Y.%m.%d")
        except Exception:
            return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _model_from_value(value: Any, models: tuple[str, ...]) -> str | None:
    token = str(value or "").strip()
    if not token:
        return None
    lowered = token.lower()
    for model in models:
        if model.lower() == lowered or model.lower() in lowered:
            return model
    return None


def _strip_price_suffix(token: str) -> str:
    parts = [part for part in token.replace("-", "_").split("_") if part]
    if len(parts) >= 2 and parts[-1].lower() in _PRICE_SUFFIXES:
        return "_".join(parts[:-1])
    return token


def normalize_forecast_symbol(
    value: Any,
    *,
    symbol_map: dict[str, str] | None = None,
    default_quote: str = "USDT",
) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    mapping = symbol_map or {}
    if raw in mapping:
        return str(mapping[raw])
    token = _strip_price_suffix(raw).upper().replace("-", "_")
    if "/" in token:
        base, quote = token.split("/", 1)
        return f"{base}/{quote}"
    if "_" in token:
        parts = [part for part in token.split("_") if part]
        if len(parts) >= 2 and len(parts[-1]) in {3, 4, 5}:
            return f"{''.join(parts[:-1])}/{parts[-1]}"
        token = "".join(parts)
    quote = str(default_quote or "").strip().upper()
    if quote and token.endswith(quote) and len(token) > len(quote):
        return f"{token[: -len(quote)]}/{quote}"
    if quote and token.isalpha() and len(token) <= 8:
        return f"{token}/{quote}"
    return token


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("predictions") or payload.get("forecasts")
        if isinstance(rows, list):
            return [dict(item) for item in rows if isinstance(item, dict)]
        return [dict(payload)]
    return []


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token:
            continue
        payload = json.loads(token)
        if isinstance(payload, dict):
            rows.append(dict(payload))
    return rows


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if pl is None:
        return []
    return [dict(row) for row in pl.read_parquet(path).to_dicts()]


def _artifact_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    suffixes = {".csv", ".json", ".jsonl", ".parquet"}
    return sorted(
        item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in suffixes
    )


class DeepLearningForecastStore:
    """Lookup DeepLearning prediction artifacts by symbol/model/time."""

    def __init__(
        self,
        path: str | Path | None,
        *,
        models: tuple[str, ...] = SUPPORTED_DEEP_LEARNING_MODELS,
        symbol_map: dict[str, str] | None = None,
        default_quote: str = "USDT",
    ) -> None:
        self.path = Path(path) if path else None
        self.models = tuple(dict.fromkeys(str(model) for model in models if str(model).strip()))
        self.symbol_map = dict(symbol_map or {})
        self.default_quote = str(default_quote or "USDT")
        self._records: dict[str, list[DeepLearningForecastRecord]] = {}
        self._origin_times: dict[str, list[datetime]] = {}
        if self.path is not None:
            self._load()

    @property
    def record_count(self) -> int:
        return sum(len(records) for records in self._records.values())

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def model_coverage(self, symbol: str) -> dict[str, int]:
        normalized = normalize_forecast_symbol(
            symbol,
            symbol_map=self.symbol_map,
            default_quote=self.default_quote,
        )
        records = self._records.get(normalized, [])
        return {
            model: sum(1 for record in records if record.model == model) for model in self.models
        }

    def snapshot(
        self,
        symbol: str,
        event_time: Any,
        *,
        current_price: float | None,
        return_threshold: float = 0.0,
        max_age_seconds: int | None = None,
        horizon_seconds: int | None = None,
    ) -> DeepLearningForecastSnapshot | None:
        event_dt = _parse_timestamp(event_time)
        if event_dt is None:
            return None
        normalized = normalize_forecast_symbol(
            symbol,
            symbol_map=self.symbol_map,
            default_quote=self.default_quote,
        )
        records = self._records.get(normalized, [])
        if not records:
            return None
        selected = self._select_records(
            normalized,
            event_dt,
            max_age_seconds=max_age_seconds,
            horizon_seconds=horizon_seconds,
        )
        if not selected:
            return None

        returns: dict[str, float] = {}
        values: dict[str, float] = {}
        confidences: dict[str, float] = {}
        for model, record in selected.items():
            pred_value = record.predicted_value
            if pred_value is not None:
                values[model] = float(pred_value)
            pred_return = record.predicted_return
            if (
                pred_return is None
                and current_price
                and current_price > 0.0
                and pred_value is not None
            ):
                pred_return = float(pred_value) / float(current_price) - 1.0
            if pred_return is None or not math.isfinite(pred_return):
                continue
            returns[model] = float(pred_return)
            if record.confidence is not None:
                confidences[model] = float(record.confidence)

        if not returns:
            return None
        arr = np.asarray(list(returns.values()), dtype=float)
        threshold = max(0.0, float(return_threshold))
        mean_return = float(np.mean(arr))
        dispersion = float(np.std(arr)) if arr.size > 1 else 0.0
        source_confidence = float(np.mean(list(confidences.values()))) if confidences else 1.0
        latest_origin = max(record.origin_time for record in selected.values())
        target_times = [record.target_time for record in selected.values() if record.target_time]
        target_time = max(target_times) if target_times else None
        return DeepLearningForecastSnapshot(
            symbol=normalized,
            event_time=event_dt,
            origin_time=latest_origin,
            target_time=target_time,
            model_returns=returns,
            model_values=values,
            model_confidences=confidences,
            mean_return=mean_return,
            dispersion=dispersion,
            long_vote_fraction=float(np.count_nonzero(arr >= threshold)) / float(arr.size),
            short_vote_fraction=float(np.count_nonzero(arr <= -threshold)) / float(arr.size),
            source_confidence=max(0.0, min(1.0, source_confidence)),
        )

    def _load(self) -> None:
        assert self.path is not None
        rows: list[tuple[Path, dict[str, Any]]] = []
        for file_path in _artifact_files(self.path):
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".csv":
                    file_rows = _read_csv_rows(file_path)
                elif suffix == ".jsonl":
                    file_rows = _read_jsonl_rows(file_path)
                elif suffix == ".json":
                    file_rows = _read_json_rows(file_path)
                elif suffix == ".parquet":
                    file_rows = _read_parquet_rows(file_path)
                else:
                    file_rows = []
            except Exception:
                file_rows = []
            rows.extend((file_path, row) for row in file_rows)
        for file_path, row in rows:
            record = self._record_from_row(row, source_path=file_path)
            if record is None:
                continue
            self._records.setdefault(record.symbol, []).append(record)
        for symbol, records in self._records.items():
            records.sort(key=lambda item: item.origin_time)
            self._origin_times[symbol] = [record.origin_time for record in records]

    def _record_from_row(
        self,
        row: dict[str, Any],
        *,
        source_path: Path,
    ) -> DeepLearningForecastRecord | None:
        model = _model_from_value(_first_value(row, _MODEL_COLUMNS), self.models)
        if model is None:
            model = _model_from_value(source_path.stem, self.models)
        if model is None:
            return None
        raw_symbol = _first_value(row, _SYMBOL_COLUMNS)
        symbol = normalize_forecast_symbol(
            raw_symbol,
            symbol_map=self.symbol_map,
            default_quote=self.default_quote,
        )
        if not symbol:
            return None
        origin_time = _parse_timestamp(_first_value(row, _ORIGIN_TIME_COLUMNS))
        target_time = _parse_timestamp(_first_value(row, _TARGET_TIME_COLUMNS))
        if origin_time is None:
            origin_time = target_time
        if origin_time is None:
            return None
        pred_return = _safe_float(_first_value(row, _RETURN_COLUMNS))
        if pred_return is None:
            return_bps = _safe_float(_first_value(row, _RETURN_BPS_COLUMNS))
            if return_bps is not None:
                pred_return = return_bps / 10_000.0
        confidence = _safe_float(_first_value(row, _CONFIDENCE_COLUMNS))
        if confidence is not None and confidence > 1.0:
            confidence = confidence / 100.0 if confidence <= 100.0 else 1.0
        predicted_value = _safe_float(_first_value(row, _VALUE_COLUMNS))
        return DeepLearningForecastRecord(
            symbol=symbol,
            model=model,
            origin_time=origin_time,
            target_time=target_time,
            predicted_value=predicted_value,
            predicted_return=pred_return,
            confidence=confidence,
            source_path=str(source_path),
        )

    def _select_records(
        self,
        symbol: str,
        event_time: datetime,
        *,
        max_age_seconds: int | None,
        horizon_seconds: int | None,
    ) -> dict[str, DeepLearningForecastRecord]:
        records = self._records.get(symbol, [])
        origins = self._origin_times.get(symbol, [])
        if not records or not origins:
            return {}
        right = bisect_right(origins, event_time)
        if right <= 0:
            return {}
        max_age = (
            None if max_age_seconds is None else timedelta(seconds=max(0, int(max_age_seconds)))
        )
        target_goal = (
            event_time + timedelta(seconds=max(0, int(horizon_seconds)))
            if horizon_seconds is not None and int(horizon_seconds) > 0
            else None
        )
        selected: dict[str, DeepLearningForecastRecord] = {}
        for model in self.models:
            model_records = [record for record in records[:right] if record.model == model]
            if not model_records:
                continue
            latest_origin = max(record.origin_time for record in model_records)
            if max_age is not None and event_time - latest_origin > max_age:
                continue
            candidates = [record for record in model_records if record.origin_time == latest_origin]
            if target_goal is None:
                future = [
                    record
                    for record in candidates
                    if record.target_time and record.target_time >= event_time
                ]
                selected[model] = min(
                    future or candidates,
                    key=lambda record: record.target_time or record.origin_time,
                )
            else:
                selected[model] = min(
                    candidates,
                    key=lambda record: abs(
                        ((record.target_time or record.origin_time) - target_goal).total_seconds()
                    ),
                )
        return selected


__all__ = [
    "SUPPORTED_DEEP_LEARNING_MODELS",
    "DeepLearningForecastRecord",
    "DeepLearningForecastSnapshot",
    "DeepLearningForecastStore",
    "normalize_deep_learning_models",
    "normalize_forecast_symbol",
]
