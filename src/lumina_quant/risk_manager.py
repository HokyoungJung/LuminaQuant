from lumina_quant.services.portfolio import PortfolioSizingService


class RiskManager:
    """Enforces risk limits before orders are sent to the exchange."""

    def __init__(self, config):
        self.config = config
        self.max_order_value = getattr(config, "MAX_ORDER_VALUE", 5000.0)
        self.max_order_notional_pct = getattr(config, "MAX_ORDER_NOTIONAL_PCT", 0.0)
        self.max_daily_loss = getattr(config, "MAX_DAILY_LOSS_PCT", 0.05)
        self.max_intraday_drawdown_pct = getattr(
            config,
            "MAX_INTRADAY_DRAWDOWN_PCT",
            self.max_daily_loss,
        )
        # Hard de-risk tier above the soft FREEZE threshold: when > 0 and intraday
        # drawdown exceeds it, force FLATTEN even if auto_flatten_on_breach is False.
        # 0.0 disables the tier (legacy behavior).
        self.hard_drawdown_flatten_pct = float(
            getattr(config, "HARD_DRAWDOWN_FLATTEN_PCT", 0.0) or 0.0
        )
        self.max_rolling_loss_pct_1h = getattr(config, "MAX_ROLLING_LOSS_PCT_1H", 0.05)
        self.max_symbol_exposure_pct = getattr(config, "MAX_SYMBOL_EXPOSURE_PCT", 0.25)
        self.max_total_margin_pct = getattr(config, "MAX_TOTAL_MARGIN_PCT", 0.5)
        max_total_notional_pct = getattr(config, "MAX_TOTAL_NOTIONAL_PCT", 0.0)
        self.max_total_notional_pct = (
            float(max_total_notional_pct)
            if float(max_total_notional_pct) > 0.0
            else self.max_total_margin_pct
        )
        self.target_allocation_mode = PortfolioSizingService.normalize_target_allocation_mode(
            getattr(config, "TARGET_ALLOCATION_MODE", "legacy_notional_cap")
        )
        self.leverage = max(1.0, float(getattr(config, "LEVERAGE", 1.0) or 1.0))
        self.freeze_new_entries_on_breach = bool(
            getattr(config, "FREEZE_NEW_ENTRIES_ON_BREACH", True)
        )
        self.auto_flatten_on_breach = bool(getattr(config, "AUTO_FLATTEN_ON_BREACH", False))
        # Phase 5 kill-switch envelope: consecutive-loss halt trigger
        self.consecutive_loss_halt_count = max(
            1, int(getattr(config, "CONSECUTIVE_LOSS_HALT_COUNT", 5))
        )
        self._consecutive_loss_count: int = 0

    @staticmethod
    def _portfolio_position_legs(portfolio, symbol) -> dict[str, float] | None:
        if portfolio is None:
            return None
        for attr in ("current_position_legs", "position_legs"):
            payload = getattr(portfolio, attr, None)
            if not isinstance(payload, dict):
                continue
            legs = payload.get(symbol)
            if isinstance(legs, dict):
                return {
                    "LONG": max(0.0, float(legs.get("LONG", 0.0) or 0.0)),
                    "SHORT": max(0.0, float(legs.get("SHORT", 0.0) or 0.0)),
                }
        return None

    @staticmethod
    def _project_hedge_legs(order_event, current_legs: dict[str, float]) -> dict[str, float] | None:
        position_side = str(getattr(order_event, "position_side", "") or "").upper()
        if position_side not in {"LONG", "SHORT"}:
            return None
        direction = str(getattr(order_event, "direction", "") or "").upper()
        quantity = abs(float(getattr(order_event, "quantity", 0.0) or 0.0))
        projected = {
            "LONG": max(0.0, float(current_legs.get("LONG", 0.0) or 0.0)),
            "SHORT": max(0.0, float(current_legs.get("SHORT", 0.0) or 0.0)),
        }
        if position_side == "LONG":
            if direction == "BUY":
                projected["LONG"] += quantity
            elif direction == "SELL":
                projected["LONG"] = max(0.0, projected["LONG"] - quantity)
        elif position_side == "SHORT":
            if direction == "SELL":
                projected["SHORT"] += quantity
            elif direction == "BUY":
                projected["SHORT"] = max(0.0, projected["SHORT"] - quantity)
        return projected

    def check_order(self, order_event, current_price, portfolio=None):
        """Returns True if order is safe, False otherwise."""
        if current_price <= 0:
            return False, "Invalid market price."

        # 1. Check Notional Value (absolute per-order cap)
        notional_value = order_event.quantity * current_price
        if float(self.max_order_value) > 0.0 and notional_value > self.max_order_value:
            return (
                False,
                f"Order Value ${notional_value:.2f} exceeds limit ${self.max_order_value}",
            )

        # 2. Check Negative Quantity
        if order_event.quantity <= 0:
            return False, f"Invalid Quantity: {order_event.quantity}"

        # 3. Portfolio-level checks
        if portfolio is not None:
            if getattr(portfolio, "trading_frozen", False) and not bool(
                getattr(order_event, "reduce_only", False)
            ):
                return False, "Trade freeze active: new entries blocked."

            # Phase 5 kill-switch: consecutive-loss auto-halt (reduce-only exempt)
            if self._consecutive_loss_count >= self.consecutive_loss_halt_count and not bool(
                getattr(order_event, "reduce_only", False)
            ):
                return (
                    False,
                    f"Consecutive-loss halt: {self._consecutive_loss_count} losses "
                    f">= limit {self.consecutive_loss_halt_count}. Only reduce-only orders allowed.",
                )

            total_equity = float(portfolio.current_holdings.get("total", 0.0))
            if total_equity <= 0:
                return False, "Non-positive equity."

            order_notional_cap = total_equity * float(self.max_order_notional_pct)
            if order_notional_cap > 0 and notional_value > order_notional_cap:
                return (
                    False,
                    f"Order Value ${notional_value:.2f} exceeds equity-scaled cap ${order_notional_cap:.2f}",
                )

            # reduce-only orders bypass circuit-breaker and all downstream exposure caps
            # so the system can always de-risk open positions after a daily-loss trip.
            # (trade-freeze and consecutive-loss halt already exempt reduce-only above.)
            if bool(getattr(order_event, "reduce_only", False)):
                return True, "Passed (reduce-only bypass)."

            if getattr(portfolio, "circuit_breaker_tripped", False):
                return False, "Circuit breaker already tripped."

            # Approximate symbol exposure after order execution.
            current_legs = self._portfolio_position_legs(portfolio, order_event.symbol)
            projected_legs = (
                self._project_hedge_legs(order_event, current_legs) if current_legs else None
            )
            if projected_legs is not None:
                projected_symbol_notional = (
                    float(projected_legs["LONG"]) + float(projected_legs["SHORT"])
                ) * current_price
            else:
                cur_qty = float(portfolio.current_positions.get(order_event.symbol, 0.0))
                signed_order_qty = (
                    order_event.quantity
                    if order_event.direction == "BUY"
                    else -order_event.quantity
                )
                projected_qty = cur_qty + signed_order_qty
                projected_symbol_notional = abs(projected_qty * current_price)
            symbol_cap = total_equity * self.max_symbol_exposure_pct
            if symbol_cap > 0 and projected_symbol_notional > symbol_cap:
                return (
                    False,
                    f"Symbol exposure {projected_symbol_notional:.2f} exceeds cap {symbol_cap:.2f}",
                )

            # Approximate total notional exposure.
            # Uses available holdings valuation from portfolio snapshot.
            current_total_notional = 0.0
            for sym in portfolio.symbol_list:
                sym_mv = float(portfolio.current_holdings.get(sym, 0.0))
                current_total_notional += abs(sym_mv)
            projected_total_notional = current_total_notional + notional_value
            total_cap = total_equity * self.max_total_notional_pct
            if total_cap > 0 and projected_total_notional > total_cap:
                return (
                    False,
                    f"Total exposure {projected_total_notional:.2f} exceeds cap {total_cap:.2f}",
                )

        return True, "Passed"

    def evaluate_portfolio_risk(self, portfolio):
        equity = float(portfolio.current_holdings.get("total", 0.0))
        if equity <= 0:
            return False, "Non-positive equity", "FLATTEN", {"equity": equity}

        day_start = float(getattr(portfolio, "day_start_equity", equity) or equity)
        intraday_loss_pct = 0.0
        if day_start > 0:
            intraday_loss_pct = max(0.0, (day_start - equity) / day_start)

        rolling_loss_pct_1h = 0.0
        get_rolling_loss = getattr(portfolio, "get_rolling_loss_pct", None)
        if callable(get_rolling_loss):
            rolling_loss_pct_1h = float(get_rolling_loss(3600))

        total_notional = 0.0
        for sym in portfolio.symbol_list:
            total_notional += abs(float(portfolio.current_holdings.get(sym, 0.0)))
        if self.target_allocation_mode == "isolated_margin_fraction":
            margin_utilization = total_notional / (equity * self.leverage) if equity > 0 else 0.0
        else:
            margin_utilization = total_notional / equity if equity > 0 else 0.0

        # Hard de-risk tier (defense-in-depth): a drawdown beyond hard_drawdown_flatten_pct
        # forces FLATTEN regardless of auto_flatten_on_breach. Evaluated before the soft
        # intraday check so the hard tier wins. 0.0 disables the tier (legacy).
        if (
            self.hard_drawdown_flatten_pct > 0.0
            and intraday_loss_pct >= self.hard_drawdown_flatten_pct
        ):
            return (
                False,
                "Hard drawdown flatten breach",
                "FLATTEN",
                {
                    "intraday_loss_pct": intraday_loss_pct,
                    "threshold": float(self.hard_drawdown_flatten_pct),
                },
            )

        if intraday_loss_pct >= float(self.max_intraday_drawdown_pct):
            action = "FLATTEN" if self.auto_flatten_on_breach else "FREEZE"
            return (
                False,
                "Intraday drawdown breach",
                action,
                {
                    "intraday_loss_pct": intraday_loss_pct,
                    "threshold": float(self.max_intraday_drawdown_pct),
                },
            )

        # Defense-in-depth daily-loss cap: mirrors the intraday-DD / rolling-loss handling.
        # Typically a looser threshold than the intraday cap, so it only fires when the
        # intraday cap is configured higher (or absent). FLATTEN if auto_flatten_on_breach
        # else FREEZE.
        if intraday_loss_pct >= float(self.max_daily_loss):
            action = "FLATTEN" if self.auto_flatten_on_breach else "FREEZE"
            return (
                False,
                "Daily loss breach",
                action,
                {
                    "intraday_loss_pct": intraday_loss_pct,
                    "threshold": float(self.max_daily_loss),
                },
            )

        if rolling_loss_pct_1h >= float(self.max_rolling_loss_pct_1h):
            action = "FLATTEN" if self.auto_flatten_on_breach else "FREEZE"
            return (
                False,
                "Rolling 1h loss breach",
                action,
                {
                    "rolling_loss_pct_1h": rolling_loss_pct_1h,
                    "threshold": float(self.max_rolling_loss_pct_1h),
                },
            )

        if margin_utilization >= float(self.max_total_margin_pct):
            action = "FREEZE" if self.freeze_new_entries_on_breach else "NONE"
            return (
                action == "NONE",
                "Margin utilization breach",
                action,
                {
                    "margin_utilization": margin_utilization,
                    "threshold": float(self.max_total_margin_pct),
                    "target_allocation_mode": self.target_allocation_mode,
                    "leverage": self.leverage,
                },
            )

        return (
            True,
            "Passed",
            "NONE",
            {
                "intraday_loss_pct": intraday_loss_pct,
                "rolling_loss_pct_1h": rolling_loss_pct_1h,
                "margin_utilization": margin_utilization,
                "target_allocation_mode": self.target_allocation_mode,
                "leverage": self.leverage,
            },
        )

    def check_portfolio_risk(self, portfolio):
        """Check if daily loss limit is hit."""
        # Already handled in Portfolio circuit breaker, but can add redundancy here.
        return True, "Passed"

    # Phase 5 kill-switch: consecutive-loss tracking
    def record_loss(self, *, realized_pnl: float) -> int:
        """Record a realized loss.  Increments the consecutive-loss counter when
        ``realized_pnl < 0``; resets it on a profitable fill.  Returns the updated
        consecutive-loss count so callers can log or audit it.
        """
        if float(realized_pnl) < 0.0:
            self._consecutive_loss_count += 1
        else:
            self._consecutive_loss_count = 0
        return self._consecutive_loss_count

    def reset_consecutive_losses(self) -> None:
        """Manually reset the consecutive-loss counter (e.g. after operator review)."""
        self._consecutive_loss_count = 0
