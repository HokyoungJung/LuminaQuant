"""Paper/testnet live adapter for the frozen Alpha Zoo Optuna hybrid."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.strategy import Strategy

from .optuna_hybrid_config import (
    DEFAULT_69_ASSET_EFFICIENCY_REPAIR_ARTIFACT,
    DEFAULT_INTEGER_PORTFOLIO_ARTIFACT,
    DEFAULT_OPTUNA_HYBRID_ARTIFACT,
    DEFAULT_SELECTED_PROFILE_ID,
    EFFICIENCY_REPAIR_FILTER_LABEL,
    INTRABAR_ATR_LOOKBACK,
    RETURN_PER_TURNOVER_THRESHOLD_BPS,
    ROUND_TRIP_COST_BPS,
    AlphaZooOptunaHybridLiveConfig,
    AlphaZooV35HybridAllocator,
    IntrabarProtectionPlan,
    SourceProfile,
    SourceSleeve,
    SleeveDecision,
    _compact_symbol,
    _live_symbol,
    load_alpha_zoo_optuna_hybrid_live_config,
)
from .optuna_hybrid_signals import (
    _build_panel,
    _clamp_stop_distance_pct,
    _evaluate_booster,
    _evaluate_cross_sectional_momentum_rank,
    _evaluate_debounced,
    _evaluate_residual,
    _evaluate_trend_pullback_reclaim,
    _evaluate_voladj,
    _evaluate_voladj_efficiency_repair,
    _frame_for,
    _intrabar_risk_frame_for,
    _latest_atr_pct,
    completed_bars_only,
    debounced_state_signal,
    trailing_state_signal,
)


class AlphaZooOptunaHybridLiveStrategy(Strategy):
    """Live-compatible paper/testnet adapter for the frozen Alpha Zoo hybrid."""

    strategy_id = "alpha_zoo_optuna_hybrid_live"
    decision_cadence_seconds = 3600
    preferred_contract = "market_window"
    uses_timeframe_aggregator = True
    required_timeframes = ("1m", "5m", "30m", "1h", "2h", "4h")
    required_lookbacks = {"1m": 96, "5m": 96, "30m": 256, "1h": 256, "2h": 256, "4h": 256}
    required_inputs = ("OHLCV",)
    strategy_validity = {
        "pass": True,
        "primary_signal_type": "artifact_frozen_state_rules",
        "causal_state_only": True,
        "lookahead_safe": True,
        "locked_oos_role": "gate_report_only",
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "rejection_reasons": ["paper_testnet_only_requires_live_fill_telemetry_before_real"],
    }

    def __init__(
        self,
        bars: Any,
        events: Any,
        *,
        optuna_hybrid_artifact_path: str | Path = DEFAULT_OPTUNA_HYBRID_ARTIFACT,
        integer_portfolio_artifact_path: str | Path = DEFAULT_INTEGER_PORTFOLIO_ARTIFACT,
        selected_profile_id: str = DEFAULT_SELECTED_PROFILE_ID,
        paper_testnet_only: bool = True,
        allow_real_money: bool = False,
        min_completed_bars: int = 0,
        **_: Any,
    ) -> None:
        if not paper_testnet_only or allow_real_money:
            raise ValueError("AlphaZooOptunaHybridLiveStrategy is paper/testnet-only")
        self.bars = bars
        self.events = events
        self.config = load_alpha_zoo_optuna_hybrid_live_config(
            optuna_hybrid_artifact_path=optuna_hybrid_artifact_path,
            integer_portfolio_artifact_path=integer_portfolio_artifact_path,
            selected_profile_id=selected_profile_id,
        )
        self.allocator = AlphaZooV35HybridAllocator(self.config)
        self.symbol_list = list(getattr(bars, "symbol_list", []) or list(self.config.watch_symbols))
        self.paper_testnet_only = True
        self.ready_for_real = False
        self.real_money_execution = False
        self.real_execution_allowed = False
        self.min_completed_bars = max(0, int(min_completed_bars))
        self._last_completed_key_by_sleeve = {
            sleeve.model_id: "" for sleeve in self.config.source_sleeves
        }
        self._last_signal_by_sleeve = {sleeve.model_id: 0 for sleeve in self.config.source_sleeves}
        self._intrabar_guards: dict[str, dict[str, Any]] = {}

    def get_state(self) -> dict[str, Any]:
        return {
            "last_completed_key_by_sleeve": dict(self._last_completed_key_by_sleeve),
            "last_signal_by_sleeve": dict(self._last_signal_by_sleeve),
            "intrabar_guards": dict(self._intrabar_guards),
            "selected_profile_id": self.config.selected_profile_id,
            "paper_testnet_only": True,
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        for key, value in dict(state.get("last_completed_key_by_sleeve") or {}).items():
            if key in self._last_completed_key_by_sleeve:
                self._last_completed_key_by_sleeve[key] = str(value)
        for key, value in dict(state.get("last_signal_by_sleeve") or {}).items():
            if key in self._last_signal_by_sleeve:
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed in {-1, 0, 1}:
                    self._last_signal_by_sleeve[key] = parsed
        guards = dict(state.get("intrabar_guards") or {})
        self._intrabar_guards = {
            str(key): dict(value)
            for key, value in guards.items()
            if isinstance(value, dict) and key in self._last_signal_by_sleeve
        }

    def calculate_signals(self, event: Any) -> None:
        if getattr(event, "type", None) != "MARKET":
            return
        self._check_intrabar_guards(event)

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        if aggregator is None:
            return
        eval_cache: dict[tuple[Any, ...], Any] = {}
        for sleeve in self.config.source_sleeves:
            decision = self._evaluate_sleeve(aggregator, sleeve, eval_cache)
            if decision is None:
                continue
            if decision.completed_key == self._last_completed_key_by_sleeve.get(sleeve.model_id):
                continue
            self._last_completed_key_by_sleeve[sleeve.model_id] = decision.completed_key
            previous_signal = int(self._last_signal_by_sleeve.get(sleeve.model_id, 0))
            if int(decision.signal) == previous_signal:
                continue
            self._last_signal_by_sleeve[sleeve.model_id] = int(decision.signal)
            self._emit_transition(sleeve, decision, previous_signal, aggregator)
        _ = event

    def _panel_for(
        self,
        aggregator: Any,
        *,
        timeframe: str,
        lookback: int,
        eval_cache: dict[tuple[Any, ...], Any],
    ) -> Any:
        symbols = tuple(self.config.watch_symbols)
        key = ("panel", timeframe, int(lookback), symbols)
        if key not in eval_cache:
            eval_cache[key] = _build_panel(aggregator, timeframe, int(lookback), symbols=symbols)
        return eval_cache[key]

    def _evaluate_sleeve(
        self,
        aggregator: Any,
        sleeve: SourceSleeve,
        eval_cache: dict[tuple[Any, ...], Any] | None = None,
    ) -> SleeveDecision | None:
        eval_cache = {} if eval_cache is None else eval_cache
        lookback = max(96, sleeve.lookback + sleeve.min_hold_bars + 8, self.min_completed_bars)
        frame = _frame_for(aggregator, sleeve.symbol, sleeve.timeframe, lookback)
        if len(frame) < max(8, sleeve.lookback + 2):
            return None
        if sleeve.family == "cross_sectional_momentum_rank":
            panel = self._panel_for(
                aggregator,
                timeframe=sleeve.timeframe,
                lookback=max(96, lookback),
                eval_cache=eval_cache,
            )
            return _evaluate_cross_sectional_momentum_rank(sleeve, frame, panel)
        if sleeve.family == "trend_pullback_reclaim":
            panel = self._panel_for(
                aggregator,
                timeframe=sleeve.timeframe,
                lookback=max(256, lookback),
                eval_cache=eval_cache,
            )
            return _evaluate_trend_pullback_reclaim(sleeve, frame, panel)
        if sleeve.family == "relative_residual_reclaim":
            base = _frame_for(aggregator, sleeve.base_symbol, sleeve.timeframe, lookback)
            panel = self._panel_for(
                aggregator,
                timeframe=sleeve.timeframe,
                lookback=max(96, lookback),
                eval_cache=eval_cache,
            )
            if len(base) < max(8, sleeve.lookback + 2):
                return None
            return _evaluate_residual(sleeve, frame, base, panel)
        if (
            sleeve.family == "volatility_adjusted_trend_persistence"
            and sleeve.filter_label == EFFICIENCY_REPAIR_FILTER_LABEL
        ):
            panel = self._panel_for(
                aggregator,
                timeframe=sleeve.timeframe,
                lookback=max(96, lookback),
                eval_cache=eval_cache,
            )
            return _evaluate_voladj_efficiency_repair(sleeve, frame, panel)
        btc = _frame_for(aggregator, "BTCUSDT", sleeve.timeframe, lookback)
        if len(btc) < max(8, sleeve.lookback + 2):
            return None
        if sleeve.family == "debounced_momentum_hysteresis_efficiency_repair":
            return _evaluate_debounced(sleeve, frame, btc)
        if sleeve.family == "relative_strength_chandelier_breakout":
            return _evaluate_booster(sleeve, frame, btc)
        if sleeve.family == "volatility_adjusted_trend_persistence":
            return _evaluate_voladj(sleeve, frame, btc)
        return None

    def target_notional_fraction_for_sleeve(self, sleeve: SourceSleeve) -> float:
        profile_weights = self.allocator.profile_weights_for_live()
        total = 0.0
        uses_weighted_notional = False
        for profile in self.config.source_profiles:
            if sleeve.model_id not in profile.selected_model_ids:
                continue
            if sleeve.weighted_notional_fraction > 0.0:
                uses_weighted_notional = True
                total += float(profile_weights.get(profile.profile_id, 0.0)) * float(
                    sleeve.weighted_notional_fraction
                )
                continue
            leverage = int(profile.leverage_map.get(sleeve.symbol, 0))
            if leverage <= 0:
                continue
            total += float(profile_weights.get(profile.profile_id, 0.0)) * float(leverage)
        if uses_weighted_notional:
            return float(total)
        return float(sleeve.allocation_fraction) * total

    def max_symbol_notional_fraction(self, symbol: str) -> float:
        """Return worst-case all-sleeve notional/equity for one live symbol."""
        compact = _compact_symbol(symbol)
        return float(
            sum(
                self.target_notional_fraction_for_sleeve(sleeve)
                for sleeve in self.config.source_sleeves
                if sleeve.symbol == compact
            )
        )

    def _profile_contributions(self, sleeve: SourceSleeve) -> list[dict[str, Any]]:
        weights = self.allocator.profile_weights_for_live()
        out: list[dict[str, Any]] = []
        for profile in self.config.source_profiles:
            if sleeve.model_id not in profile.selected_model_ids:
                continue
            leverage = int(profile.leverage_map.get(sleeve.symbol, 0))
            weight = float(weights.get(profile.profile_id, 0.0))
            source_notional = (
                float(sleeve.weighted_notional_fraction)
                if sleeve.weighted_notional_fraction > 0.0
                else float(sleeve.allocation_fraction) * float(leverage)
            )
            out.append(
                {
                    "profile_id": profile.profile_id,
                    "profile_weight": weight,
                    "integer_leverage": leverage,
                    "weighted_integer_leverage": weight * float(leverage),
                    "source_notional_fraction": source_notional,
                    "weighted_notional_fraction": weight * source_notional,
                }
            )
        return out

    def _build_intrabar_protection_plan(
        self,
        aggregator: Any,
        sleeve: SourceSleeve,
        decision: SleeveDecision,
        signal_type: str,
    ) -> IntrabarProtectionPlan:
        if signal_type not in {"LONG", "SHORT"}:
            return IntrabarProtectionPlan(
                enabled=False,
                source_timeframe=sleeve.timeframe,
                guard_mode="none_exit_signal",
                stop_loss=None,
                take_profit=None,
                trailing_percent=None,
                stop_distance_pct=0.0,
                atr_pct=None,
                notes=("exit signals clear any active intrabar guard",),
            )
        frame, source_timeframe = _intrabar_risk_frame_for(
            aggregator,
            sleeve.symbol,
            sleeve.timeframe,
            max(INTRABAR_ATR_LOOKBACK * 4, sleeve.lookback + 4),
        )
        atr_pct = _latest_atr_pct(frame)
        stop_pct = _clamp_stop_distance_pct(atr_pct)
        price = max(1e-12, float(decision.price))
        if signal_type == "LONG":
            stop_loss = price * (1.0 - stop_pct)
        else:
            stop_loss = price * (1.0 + stop_pct)
        trailing_percent = stop_pct if sleeve.trail_atr_mult > 0.0 else None
        notes = (
            "paper/testnet local intrabar guard emits component EXIT on stop breach",
            "real exchange-side protective orders still require explicit exchange order support",
            "queue priority is measured as telemetry/proxy, not exactly knowable from exchange APIs",
        )
        return IntrabarProtectionPlan(
            enabled=True,
            source_timeframe=source_timeframe,
            guard_mode="paper_local_or_simulated_component_exit",
            stop_loss=float(stop_loss),
            take_profit=None,
            trailing_percent=trailing_percent,
            stop_distance_pct=stop_pct,
            atr_pct=atr_pct,
            notes=notes,
        )

    def _activate_intrabar_guard(
        self,
        sleeve: SourceSleeve,
        decision: SleeveDecision,
        signal_type: str,
        plan: IntrabarProtectionPlan,
    ) -> None:
        if not plan.enabled or plan.stop_loss is None:
            self._intrabar_guards.pop(sleeve.model_id, None)
            return
        self._intrabar_guards[sleeve.model_id] = {
            "source_model_id": sleeve.model_id,
            "component_id": sleeve.model_id,
            "symbol": _live_symbol(sleeve.symbol),
            "compact_symbol": sleeve.symbol,
            "side": signal_type,
            "stop_loss": float(plan.stop_loss),
            "take_profit": None if plan.take_profit is None else float(plan.take_profit),
            "trailing_percent": None
            if plan.trailing_percent is None
            else float(plan.trailing_percent),
            "highest_price": float(decision.price),
            "lowest_price": float(decision.price),
            "source_timeframe": plan.source_timeframe,
            "activated_at": decision.completed_key,
            "entry_price": float(decision.price),
            "guard_mode": plan.guard_mode,
        }

    def _check_intrabar_guards(self, event: Any) -> None:
        symbol = _compact_symbol(getattr(event, "symbol", ""))
        if not symbol or not self._intrabar_guards:
            return
        high = float(getattr(event, "high", 0.0) or 0.0)
        low = float(getattr(event, "low", 0.0) or 0.0)
        close = float(getattr(event, "close", 0.0) or 0.0)
        if high <= 0.0 or low <= 0.0:
            return
        for model_id, guard in list(self._intrabar_guards.items()):
            if (
                _compact_symbol(str(guard.get("symbol") or guard.get("compact_symbol") or ""))
                != symbol
            ):
                continue
            side = str(guard.get("side") or "").upper()
            stop_loss = float(guard.get("stop_loss") or 0.0)
            trailing_percent = guard.get("trailing_percent")
            trigger_price = None
            reason = ""
            if side == "LONG":
                guard["highest_price"] = max(float(guard.get("highest_price") or 0.0), high)
                if trailing_percent is not None:
                    stop_loss = max(
                        stop_loss, float(guard["highest_price"]) * (1.0 - float(trailing_percent))
                    )
                    guard["stop_loss"] = stop_loss
                if low <= stop_loss:
                    trigger_price = stop_loss
                    reason = "intrabar_stop_loss_or_trailing_long"
            elif side == "SHORT":
                previous_low = float(guard.get("lowest_price") or close or low)
                guard["lowest_price"] = min(previous_low, low)
                if trailing_percent is not None:
                    stop_loss = min(
                        stop_loss, float(guard["lowest_price"]) * (1.0 + float(trailing_percent))
                    )
                    guard["stop_loss"] = stop_loss
                if high >= stop_loss:
                    trigger_price = stop_loss
                    reason = "intrabar_stop_loss_or_trailing_short"
            if trigger_price is None:
                continue
            self._intrabar_guards.pop(model_id, None)
            self._last_signal_by_sleeve[model_id] = 0
            client_hash = hashlib.sha1(
                f"{model_id}|{getattr(event, 'time', '')}|{reason}".encode()
            ).hexdigest()[:18]
            locked_oos_role = str(self.config.governance.get("locked_oos_role", "gate_report_only"))
            self.events.put(
                SignalEvent(
                    strategy_id=self.strategy_id,
                    symbol=str(guard.get("symbol")),
                    datetime=getattr(event, "time", None),
                    signal_type="EXIT",
                    strength=1.0,
                    price=float(trigger_price),
                    position_side=side,
                    client_order_id=f"azoh-risk-{client_hash}",
                    metadata={
                        "alpha_zoo_optuna_hybrid_live": True,
                        "paper_testnet_only": True,
                        "ready_for_real": False,
                        "real_money_execution": False,
                        "real_execution_allowed": False,
                        "component_id": model_id,
                        "source_model_id": model_id,
                        "intrabar_protection_enabled": True,
                        "intrabar_exit_reason": reason,
                        "intrabar_trigger_price": float(trigger_price),
                        "intrabar_event_high": high,
                        "intrabar_event_low": low,
                        "intrabar_event_close": close,
                        "target_allocation": 0.0,
                        "target_allocation_mode": "notional_fraction",
                        "sizing_mode": "notional_fraction",
                        "locked_oos_role": locked_oos_role,
                        "live_unfilled_order_policy": dict(
                            self.config.governance.get("live_unfilled_order_policy") or {}
                        ),
                        "live_slippage_guard_policy": dict(
                            self.config.governance.get("live_slippage_guard_policy") or {}
                        ),
                    },
                )
            )

    def _emit_transition(
        self,
        sleeve: SourceSleeve,
        decision: SleeveDecision,
        previous_signal: int,
        aggregator: Any,
    ) -> None:
        if int(decision.signal) > 0:
            signal_type = "LONG"
            position_side = "LONG"
        elif int(decision.signal) < 0:
            signal_type = "SHORT"
            position_side = "SHORT"
        else:
            signal_type = "EXIT"
            position_side = (
                "LONG" if previous_signal > 0 else "SHORT" if previous_signal < 0 else None
            )
        protection = self._build_intrabar_protection_plan(aggregator, sleeve, decision, signal_type)
        if signal_type == "EXIT":
            self._intrabar_guards.pop(sleeve.model_id, None)
        target_notional = self.target_notional_fraction_for_sleeve(sleeve)
        locked_oos_role = str(self.config.governance.get("locked_oos_role", "gate_report_only"))
        target_formula = (
            "sum(profile_weight*source_notional_fraction_after_sleeve_multiplier)"
            if sleeve.weighted_notional_fraction > 0.0
            else "allocation_fraction*sum(profile_weight*integer_leverage)"
        )
        symbol_notional_cap = max(
            target_notional + 0.05,
            self.max_symbol_notional_fraction(sleeve.symbol) * 1.05,
        )
        metadata = {
            "alpha_zoo_optuna_hybrid_live": True,
            "paper_testnet_only": True,
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
            "source_model_id": sleeve.model_id,
            "source_family": sleeve.family,
            "source_symbol": sleeve.symbol,
            "source_timeframe": sleeve.timeframe,
            "source_side": sleeve.side,
            "component_id": sleeve.model_id,
            "source_allocation_fraction": float(sleeve.allocation_fraction),
            "source_research_leverage": float(sleeve.source_leverage),
            "source_weighted_notional_fraction": float(sleeve.weighted_notional_fraction),
            "source_sleeve_multiplier": float(sleeve.sleeve_multiplier),
            "target_notional_fraction": target_notional,
            "target_notional_formula": target_formula,
            "target_allocation": target_notional,
            "target_allocation_mode": "notional_fraction",
            "sizing_mode": "notional_fraction",
            "max_order_value": 0.0,
            "max_order_notional_pct": max(target_notional + 0.05, target_notional * 1.05),
            "max_symbol_exposure_pct": symbol_notional_cap,
            "profile_contributions": self._profile_contributions(sleeve),
            "allocator": self.allocator.allocation_metadata(),
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "return_per_turnover_threshold_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
            "locked_oos_role": locked_oos_role,
            "replay_live_notional_parity": True,
            "completed_bar_key": decision.completed_key,
            "previous_source_signal": int(previous_signal),
            "current_source_signal": int(decision.signal),
            "diagnostics": decision.diagnostics,
            "intrabar_protection_enabled": protection.enabled,
            "intrabar_protection": {
                "guard_mode": protection.guard_mode,
                "source_timeframe": protection.source_timeframe,
                "stop_loss": protection.stop_loss,
                "take_profit": protection.take_profit,
                "trailing_percent": protection.trailing_percent,
                "stop_distance_pct": protection.stop_distance_pct,
                "atr_pct": protection.atr_pct,
                "notes": list(protection.notes),
            },
            "microstructure_telemetry_required": [
                "bbo_spread_bps_at_submit",
                "order_book_depth_or_liquidity_proxy",
                "submit_to_fill_ms",
                "realized_slippage_bps",
                "fee_bps",
                "funding_bps",
                "partial_fill_ratio",
                "cancel_timeout_reject_flags",
            ],
            "queue_priority_model": "proxy_only_exchange_exact_queue_position_unavailable",
            "live_unfilled_order_policy": dict(
                self.config.governance.get("live_unfilled_order_policy") or {}
            ),
            "live_slippage_guard_policy": dict(
                self.config.governance.get("live_slippage_guard_policy") or {}
            ),
        }
        if signal_type in {"LONG", "SHORT"}:
            self._activate_intrabar_guard(sleeve, decision, signal_type, protection)
        client_hash = hashlib.sha1(
            f"{sleeve.model_id}|{decision.completed_key}|{signal_type}".encode()
        ).hexdigest()[:18]
        self.events.put(
            SignalEvent(
                strategy_id=self.strategy_id,
                symbol=_live_symbol(sleeve.symbol),
                datetime=decision.event_time,
                signal_type=signal_type,
                strength=abs(target_notional),
                price=float(decision.price),
                stop_loss=protection.stop_loss,
                take_profit=protection.take_profit,
                position_side=position_side,
                client_order_id=f"azoh-{client_hash}",
                metadata=metadata,
                trailing_percent=protection.trailing_percent,
            )
        )


__all__ = [
    "DEFAULT_69_ASSET_EFFICIENCY_REPAIR_ARTIFACT",
    "DEFAULT_INTEGER_PORTFOLIO_ARTIFACT",
    "DEFAULT_OPTUNA_HYBRID_ARTIFACT",
    "DEFAULT_SELECTED_PROFILE_ID",
    "RETURN_PER_TURNOVER_THRESHOLD_BPS",
    "ROUND_TRIP_COST_BPS",
    "AlphaZooOptunaHybridLiveConfig",
    "AlphaZooOptunaHybridLiveStrategy",
    "AlphaZooV35HybridAllocator",
    "IntrabarProtectionPlan",
    "SourceProfile",
    "SourceSleeve",
    "completed_bars_only",
    "debounced_state_signal",
    "load_alpha_zoo_optuna_hybrid_live_config",
    "trailing_state_signal",
]
