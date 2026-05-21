"""Small data models for the public sample pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class Bar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class TargetPosition(StrEnum):
    FLAT = "flat"
    LONG = "long"


@dataclass(frozen=True, slots=True)
class Signal:
    timestamp: str
    target: TargetPosition
    reason: str


@dataclass(frozen=True, slots=True)
class Trade:
    timestamp: str
    side: str
    quantity: float
    price: float
    fee: float


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: str
    equity: float
