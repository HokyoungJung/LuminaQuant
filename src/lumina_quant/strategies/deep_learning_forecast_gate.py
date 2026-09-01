"""DeepLearning artifact-driven forecast consensus strategy.

The strategy never trains or imports DeepLearning models. It only consumes saved
prediction artifacts from FITS, CycleNet, CMamba, PatchTST, or a configured subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumina_quant.market_units import BPS_PER_UNIT
from lumina_quant.core.events import SignalEvent
from lumina_quant.core.plugin_registry import register
from lumina_quant.data.deep_learning_forecasts import (
    SUPPORTED_DEEP_LEARNING_MODELS,
    DeepLearningForecastSnapshot,
    DeepLearningForecastStore,
    normalize_deep_learning_models,
)
from lumina_quant.indicators.common import safe_float
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _ForecastGateState:
    position: str = "OUT"
    last_time_key: str = ""


def _parse_models(value: Any) -> tuple[str, ...]:
    return normalize_deep_learning_models(value)


@register("strategy", "DeepLearningForecastGateStrategy", interface="event_driven")
class DeepLearningForecastGateStrategy(Strategy):
    """Trade only when saved DeepLearning model forecasts agree."""

    preferred_contract = "market_window"
    required_inputs = ("market_window",)
    required_features = ("deep_learning_forecast_artifact",)

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "forecast_path": HyperParam.string(
                "forecast_path",
                default="",
                description="Directory/file with DeepLearning prediction artifacts.",
            ),
            "models": HyperParam.string(
                "models",
                default=",".join(SUPPORTED_DEEP_LEARNING_MODELS),
                description="Comma-separated DeepLearning models to require/score.",
            ),
            "horizon_seconds": HyperParam.integer(
                "horizon_seconds",
                default=3600,
                low=0,
                high=86_400 * 30,
                tunable=False,
            ),
            "max_forecast_age_seconds": HyperParam.integer(
                "max_forecast_age_seconds",
                default=86_400,
                low=0,
                high=86_400 * 30,
                tunable=False,
            ),
            "entry_threshold_bps": HyperParam.floating(
                "entry_threshold_bps",
                default=10.0,
                low=0.0,
                high=1_000.0,
                optuna={"type": "float", "low": 2.0, "high": 80.0},
                grid=[5.0, 10.0, 20.0],
            ),
            "exit_threshold_bps": HyperParam.floating(
                "exit_threshold_bps",
                default=2.0,
                low=0.0,
                high=1_000.0,
                optuna={"type": "float", "low": 0.0, "high": 20.0},
                grid=[0.0, 2.0, 5.0],
            ),
            "min_model_agreement": HyperParam.floating(
                "min_model_agreement",
                default=0.75,
                low=0.0,
                high=1.0,
                optuna={"type": "float", "low": 0.50, "high": 1.0},
                grid=[0.50, 0.75, 1.0],
            ),
            "max_dispersion_bps": HyperParam.floating(
                "max_dispersion_bps",
                default=80.0,
                low=0.0,
                high=10_000.0,
                optuna={"type": "float", "low": 20.0, "high": 250.0},
                grid=[40.0, 80.0, 160.0],
            ),
            "min_confidence": HyperParam.floating(
                "min_confidence",
                default=0.0,
                low=0.0,
                high=1.0,
                optuna={"type": "float", "low": 0.0, "high": 0.80},
                grid=[0.0, 0.25, 0.50],
            ),
            "min_models": HyperParam.integer(
                "min_models",
                default=2,
                low=1,
                high=len(SUPPORTED_DEEP_LEARNING_MODELS),
                tunable=False,
            ),
            "allow_short": HyperParam.boolean(
                "allow_short",
                default=True,
                optuna={"type": "categorical", "choices": [True, False]},
                grid=[True, False],
            ),
            "exit_on_uncertain": HyperParam.boolean(
                "exit_on_uncertain",
                default=True,
                optuna={"type": "categorical", "choices": [True, False]},
                grid=[True, False],
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation",
                default=0.05,
                low=0.0,
                high=2.0,
                tunable=False,
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value",
                default=0.0,
                low=0.0,
                high=1_000_000.0,
                tunable=False,
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct",
                default=0.0,
                low=0.0,
                high=0.95,
                tunable=False,
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct",
                default=0.0,
                low=0.0,
                high=10.0,
                tunable=False,
            ),
            "default_quote": HyperParam.string(
                "default_quote",
                default="USDT",
                description="Quote appended when a DeepLearning dbcode has only the base token.",
            ),
        }

    def __init__(
        self,
        bars,
        events,
        forecast_path: str = "",
        models: str = ",".join(SUPPORTED_DEEP_LEARNING_MODELS),
        horizon_seconds: int = 3600,
        max_forecast_age_seconds: int = 86_400,
        entry_threshold_bps: float = 10.0,
        exit_threshold_bps: float = 2.0,
        min_model_agreement: float = 0.75,
        max_dispersion_bps: float = 80.0,
        min_confidence: float = 0.0,
        min_models: int = 2,
        allow_short: bool = True,
        exit_on_uncertain: bool = True,
        target_allocation: float = 0.05,
        max_order_value: float = 0.0,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
        default_quote: str = "USDT",
    ) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "forecast_path": forecast_path,
                "models": models,
                "horizon_seconds": horizon_seconds,
                "max_forecast_age_seconds": max_forecast_age_seconds,
                "entry_threshold_bps": entry_threshold_bps,
                "exit_threshold_bps": exit_threshold_bps,
                "min_model_agreement": min_model_agreement,
                "max_dispersion_bps": max_dispersion_bps,
                "min_confidence": min_confidence,
                "min_models": min_models,
                "allow_short": allow_short,
                "exit_on_uncertain": exit_on_uncertain,
                "target_allocation": target_allocation,
                "max_order_value": max_order_value,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "default_quote": default_quote,
            },
            keep_unknown=False,
        )
        self.models = _parse_models(resolved["models"])
        self.horizon_seconds = int(resolved["horizon_seconds"])
        self.max_forecast_age_seconds = int(resolved["max_forecast_age_seconds"])
        self.entry_threshold = float(resolved["entry_threshold_bps"]) / BPS_PER_UNIT
        self.exit_threshold = float(resolved["exit_threshold_bps"]) / BPS_PER_UNIT
        self.min_model_agreement = float(resolved["min_model_agreement"])
        self.max_dispersion = float(resolved["max_dispersion_bps"]) / BPS_PER_UNIT
        self.min_confidence = float(resolved["min_confidence"])
        self.min_models = max(1, int(resolved["min_models"]))
        self.allow_short = bool(resolved["allow_short"])
        self.exit_on_uncertain = bool(resolved["exit_on_uncertain"])
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.default_quote = str(resolved["default_quote"] or "USDT")
        self.forecasts = DeepLearningForecastStore(
            str(resolved["forecast_path"]),
            models=self.models,
            default_quote=self.default_quote,
        )
        self._state = {symbol: _ForecastGateState() for symbol in self.symbol_list}

    def get_state(self) -> dict:
        return {
            "symbol_state": {
                symbol: {
                    "position": item.position,
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        raw_state = state.get("symbol_state")
        if not isinstance(raw_state, dict):
            return
        for symbol, raw in raw_state.items():
            if symbol not in self._state or not isinstance(raw, dict):
                continue
            position = str(raw.get("position", "OUT")).upper()
            self._state[symbol].position = (
                position if position in {"OUT", "LONG", "SHORT"} else "OUT"
            )
            self._state[symbol].last_time_key = str(raw.get("last_time_key", ""))

    def calculate_signals(self, event) -> None:
        if getattr(event, "type", None) != "MARKET":
            return
        symbol_obj = getattr(event, "symbol", None)
        if symbol_obj not in self._state:
            return
        symbol = str(symbol_obj)
        item = self._state[symbol]
        event_time = getattr(event, "time", None)
        time_key = "" if event_time is None else str(event_time)
        if time_key and time_key == item.last_time_key:
            return
        if time_key:
            item.last_time_key = time_key

        price = safe_float(getattr(event, "close", None))
        if price is None:
            price = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if price is None or price <= 0.0:
            return

        snapshot = self.forecasts.snapshot(
            symbol,
            event_time,
            current_price=price,
            return_threshold=self.entry_threshold,
            max_age_seconds=self.max_forecast_age_seconds,
            horizon_seconds=self.horizon_seconds,
        )
        if snapshot is None:
            return
        if snapshot.model_count < self.min_models:
            return

        uncertain = self._is_uncertain(snapshot)
        if item.position == "OUT":
            if not uncertain and snapshot.long_vote_fraction >= self.min_model_agreement:
                self._emit(symbol, event_time, "LONG", price, snapshot)
                item.position = "LONG"
            elif (
                self.allow_short
                and not uncertain
                and snapshot.short_vote_fraction >= self.min_model_agreement
            ):
                self._emit(symbol, event_time, "SHORT", price, snapshot)
                item.position = "SHORT"
            return

        if item.position == "LONG":
            should_exit = (
                snapshot.mean_return <= self.exit_threshold
                or snapshot.short_vote_fraction >= self.min_model_agreement
                or (uncertain and self.exit_on_uncertain)
            )
            if should_exit:
                self._emit(symbol, event_time, "EXIT", price, snapshot, reason="forecast_long_exit")
                item.position = "OUT"
            return

        if item.position == "SHORT":
            should_exit = (
                snapshot.mean_return >= -self.exit_threshold
                or snapshot.long_vote_fraction >= self.min_model_agreement
                or (uncertain and self.exit_on_uncertain)
            )
            if should_exit:
                self._emit(
                    symbol, event_time, "EXIT", price, snapshot, reason="forecast_short_exit"
                )
                item.position = "OUT"

    def _is_uncertain(self, snapshot: DeepLearningForecastSnapshot) -> bool:
        if snapshot.source_confidence < self.min_confidence:
            return True
        return self.max_dispersion > 0.0 and snapshot.dispersion > self.max_dispersion

    def _conviction(self, snapshot: DeepLearningForecastSnapshot) -> float:
        agreement = max(snapshot.long_vote_fraction, snapshot.short_vote_fraction)
        threshold = max(self.entry_threshold, 1e-12)
        magnitude = min(2.0, abs(snapshot.mean_return) / threshold)
        if self.max_dispersion > 0.0:
            dispersion_penalty = max(0.0, 1.0 - snapshot.dispersion / self.max_dispersion)
        else:
            dispersion_penalty = 1.0
        return max(
            0.0,
            min(
                1.0,
                0.5 * magnitude * agreement * snapshot.source_confidence + 0.5 * dispersion_penalty,
            ),
        )

    def _emit(
        self,
        symbol: str,
        event_time: Any,
        signal_type: str,
        price: float,
        snapshot: DeepLearningForecastSnapshot,
        *,
        reason: str = "forecast_consensus",
    ) -> None:
        conviction = self._conviction(snapshot)
        target_allocation = self.target_allocation * conviction if signal_type != "EXIT" else 0.0
        metadata = {
            "strategy": "DeepLearningForecastGateStrategy",
            "reason": reason,
            "models": list(snapshot.model_returns.keys()),
            "model_returns": dict(snapshot.model_returns),
            "mean_pred_return": float(snapshot.mean_return),
            "model_dispersion": float(snapshot.dispersion),
            "long_vote_fraction": float(snapshot.long_vote_fraction),
            "short_vote_fraction": float(snapshot.short_vote_fraction),
            "source_confidence": float(snapshot.source_confidence),
            "forecast_origin_time": snapshot.origin_time.isoformat(),
            "forecast_target_time": snapshot.target_time.isoformat()
            if snapshot.target_time is not None
            else None,
            "target_allocation": float(target_allocation),
            "max_symbol_exposure_pct": float(target_allocation),
        }
        if self.max_order_value > 0.0 and signal_type != "EXIT":
            metadata["max_order_value"] = float(self.max_order_value)
        stop_loss = None
        take_profit = None
        if signal_type == "LONG":
            stop_loss = price * (1.0 - self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
            take_profit = (
                price * (1.0 + self.take_profit_pct) if self.take_profit_pct > 0.0 else None
            )
        elif signal_type == "SHORT":
            stop_loss = price * (1.0 + self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
            take_profit = (
                price * (1.0 - self.take_profit_pct) if self.take_profit_pct > 0.0 else None
            )
        self.events.put(
            SignalEvent(
                strategy_id="deep_learning_forecast_gate",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(target_allocation if target_allocation > 0.0 else 1.0),
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_side=signal_type if signal_type in {"LONG", "SHORT"} else None,
                metadata=metadata,
            )
        )


__all__ = ["DeepLearningForecastGateStrategy"]
