"""Strict latest-anchored final-validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.market_data import (
    load_data_dict_from_parquet,
    load_futures_feature_points_from_db,
)
from lumina_quant.portfolio_split_contract import OOS_START, TRAIN_START, VAL_START
from lumina_quant.storage.parquet import (
    normalize_symbol,
    normalize_timeframe_token,
    timeframe_to_milliseconds,
)
from lumina_quant.timeframe_aggregator import resample_ohlcv_frame_to_timeframe

_TIMEFRAME_ORDER = ("1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d")


@dataclass(frozen=True, slots=True)
class LoadedFrameInfo:
    symbol: str
    timeframe: str
    source_timeframe: str
    rebuilt_from_lower_timeframe: bool
    start_utc: str | None
    end_utc: str | None
    row_count: int


@dataclass(frozen=True, slots=True)
class LatestAnchoredSplit:
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    oos_start: datetime
    oos_end: datetime

    def as_dict(self) -> dict[str, str]:
        return {
            "train_start": _iso_utc(self.train_start),
            "train_end": _iso_utc(self.train_end),
            "val_start": _iso_utc(self.val_start),
            "val_end": _iso_utc(self.val_end),
            "oos_start": _iso_utc(self.oos_start),
            "oos_end": _iso_utc(self.oos_end),
        }


def required_symbol_timeframes(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    required: set[tuple[str, str]] = set()
    for row in list(rows or []):
        timeframe = normalize_timeframe_token(
            str(row.get("strategy_timeframe") or row.get("timeframe") or "1m")
        )
        symbols = [normalize_symbol(str(symbol)) for symbol in list(row.get("symbols") or [])]
        for symbol in symbols:
            if symbol:
                required.add((symbol, timeframe))
    return required


def required_feature_symbols(
    rows: list[dict[str, Any]], *, feature_required_strategies: set[str]
) -> list[str]:
    ordered: dict[str, None] = {}
    for row in list(rows or []):
        strategy_class = str(row.get("strategy_class") or row.get("strategy") or "")
        if strategy_class not in feature_required_strategies:
            continue
        for symbol in list(row.get("symbols") or []):
            token = normalize_symbol(str(symbol))
            if token:
                ordered[token] = None
    return list(ordered)


def load_real_ohlcv_frame(
    *,
    root_path: str,
    exchange: str,
    symbol: str,
    timeframe: str,
    start_date: Any,
    end_date: Any,
) -> tuple[pl.DataFrame, LoadedFrameInfo]:
    target_symbol = normalize_symbol(symbol)
    target_timeframe = normalize_timeframe_token(timeframe)

    def _load_frame(token: str) -> pl.DataFrame:
        try:
            frames = load_data_dict_from_parquet(
                root_path,
                exchange=exchange,
                symbol_list=[target_symbol],
                timeframe=token,
                start_date=start_date,
                end_date=end_date,
                data_mode="raw-first",
            )
        except RawFirstDataMissingError:
            return pl.DataFrame()
        return frames.get(target_symbol, pl.DataFrame())

    direct = _load_frame(target_timeframe)
    if not direct.is_empty():
        return direct, LoadedFrameInfo(
            symbol=target_symbol,
            timeframe=target_timeframe,
            source_timeframe=target_timeframe,
            rebuilt_from_lower_timeframe=False,
            start_utc=_frame_start_utc(direct),
            end_utc=_frame_end_utc(direct),
            row_count=int(direct.height),
        )

    target_ms = int(timeframe_to_milliseconds(target_timeframe))
    compatible_sources = [
        token
        for token in _TIMEFRAME_ORDER
        if token != target_timeframe
        and int(timeframe_to_milliseconds(token)) < target_ms
        and (target_ms % int(timeframe_to_milliseconds(token)) == 0)
    ]
    compatible_sources.sort(key=lambda token: int(timeframe_to_milliseconds(token)), reverse=True)
    for source_timeframe in compatible_sources:
        source_frame = _load_frame(source_timeframe)
        if source_frame.is_empty():
            continue
        rebuilt = resample_ohlcv_frame_to_timeframe(
            source_frame,
            source_timeframe=source_timeframe,
            timeframe=target_timeframe,
            drop_incomplete_last=True,
        )
        if rebuilt.is_empty():
            continue
        return rebuilt, LoadedFrameInfo(
            symbol=target_symbol,
            timeframe=target_timeframe,
            source_timeframe=source_timeframe,
            rebuilt_from_lower_timeframe=True,
            start_utc=_frame_start_utc(rebuilt),
            end_utc=_frame_end_utc(rebuilt),
            row_count=int(rebuilt.height),
        )

    raise RawFirstDataMissingError(
        f"No real committed OHLCV available for {target_symbol}:{target_timeframe}; no lower-timeframe rebuild source found."
    )


def discover_latest_common_complete_timestamp(
    *,
    root_path: str,
    exchange: str,
    rows: list[dict[str, Any]],
    feature_symbols: list[str],
    suite_start: datetime,
    requested_end: Any = None,
) -> tuple[datetime, list[LoadedFrameInfo]]:
    infos: list[LoadedFrameInfo] = []
    end_candidates: list[datetime] = []
    for symbol, timeframe in sorted(required_symbol_timeframes(rows)):
        frame, info = load_real_ohlcv_frame(
            root_path=root_path,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=suite_start,
            end_date=requested_end,
        )
        infos.append(info)
        end_dt = _frame_end_datetime(frame)
        if end_dt is None:
            raise RawFirstDataMissingError(
                f"Real OHLCV frame has no end timestamp for {symbol}:{timeframe}."
            )
        end_candidates.append(end_dt)

    for symbol in list(feature_symbols or []):
        feature_frame = load_futures_feature_points_from_db(
            root_path,
            exchange=exchange,
            symbol=normalize_symbol(symbol),
            start_date=suite_start,
            end_date=requested_end,
        )
        if feature_frame.is_empty() or "datetime" not in feature_frame.columns:
            raise RawFirstDataMissingError(
                f"Real futures feature points missing for {normalize_symbol(symbol)}."
            )
        end_dt = _frame_end_datetime(feature_frame)
        if end_dt is None:
            raise RawFirstDataMissingError(
                f"Real futures feature frame has no end timestamp for {normalize_symbol(symbol)}."
            )
        end_candidates.append(end_dt)

    if not end_candidates:
        raise RawFirstDataMissingError(
            "No real-data end timestamps were available for final validation."
        )
    return min(end_candidates), infos


def build_latest_anchored_split(
    *,
    saved_oos_end: datetime,
    anchored_oos_end: datetime,
) -> LatestAnchoredSplit:
    canonical_train_start = _parse_utc(TRAIN_START)
    canonical_val_start = _parse_utc(VAL_START)
    canonical_oos_start = _parse_utc(OOS_START)
    train_duration = canonical_val_start - canonical_train_start
    val_duration = canonical_oos_start - canonical_val_start
    oos_duration = saved_oos_end - canonical_oos_start

    oos_end = anchored_oos_end
    oos_start = oos_end - oos_duration
    val_start = oos_start - val_duration
    train_start = val_start - train_duration
    train_end = val_start
    val_end = oos_start
    return LatestAnchoredSplit(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        oos_start=oos_start,
        oos_end=oos_end,
    )


def load_final_validation_cache(
    *,
    root_path: str,
    exchange: str,
    rows: list[dict[str, Any]],
    split: LatestAnchoredSplit,
) -> tuple[dict[tuple[str, str], Any], list[LoadedFrameInfo]]:
    cache: dict[tuple[str, str], Any] = {}
    infos: list[LoadedFrameInfo] = []
    requirements = required_symbol_timeframes(rows)
    for _symbol, timeframe in list(requirements):
        requirements.add(("BTC/USDT", timeframe))
    for symbol, timeframe in sorted(requirements):
        frame, info = load_real_ohlcv_frame(
            root_path=root_path,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=split.train_start,
            end_date=split.oos_end,
        )
        infos.append(info)
        from lumina_quant.strategy_factory.research_runner import SeriesBundle

        sorted_frame = frame.sort("datetime")
        cache[(normalize_symbol(symbol), normalize_timeframe_token(timeframe))] = SeriesBundle(
            symbol=normalize_symbol(symbol),
            timeframe=normalize_timeframe_token(timeframe),
            datetime=sorted_frame["datetime"].to_numpy(),
            open=sorted_frame["open"].to_numpy(),
            high=sorted_frame["high"].to_numpy(),
            low=sorted_frame["low"].to_numpy(),
            close=sorted_frame["close"].to_numpy(),
            volume=sorted_frame["volume"].to_numpy(),
        )
    return cache, infos


def _frame_start_utc(frame: pl.DataFrame) -> str | None:
    dt = _frame_start_datetime(frame)
    return _iso_utc(dt) if dt is not None else None


def _frame_end_utc(frame: pl.DataFrame) -> str | None:
    dt = _frame_end_datetime(frame)
    return _iso_utc(dt) if dt is not None else None


def _frame_start_datetime(frame: pl.DataFrame) -> datetime | None:
    if frame.is_empty() or "datetime" not in frame.columns:
        return None
    value = frame["datetime"].min()
    return _coerce_datetime(value)


def _frame_end_datetime(frame: pl.DataFrame) -> datetime | None:
    if frame.is_empty() or "datetime" not in frame.columns:
        return None
    value = frame["datetime"].max()
    return _coerce_datetime(value)


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if hasattr(value, "to_pydatetime"):
        converted = value.to_pydatetime()
        return converted.astimezone(UTC) if converted.tzinfo else converted.replace(tzinfo=UTC)
    try:
        converted = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    return converted.astimezone(UTC) if converted.tzinfo else converted.replace(tzinfo=UTC)


def _parse_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    token = str(value or "").strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(token)
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "LatestAnchoredSplit",
    "LoadedFrameInfo",
    "build_latest_anchored_split",
    "discover_latest_common_complete_timestamp",
    "load_final_validation_cache",
    "required_feature_symbols",
    "required_symbol_timeframes",
]
