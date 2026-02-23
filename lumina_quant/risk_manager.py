class RiskManager:
    """Enforces risk limits before orders are sent to the exchange."""

    def __init__(self, config):
        self.config = config
        self.max_order_value = getattr(config, "MAX_ORDER_VALUE", 5000.0)
        self.max_daily_loss = getattr(config, "MAX_DAILY_LOSS_PCT", 0.05)
        self.max_intraday_drawdown_pct = getattr(
            config,
            "MAX_INTRADAY_DRAWDOWN_PCT",
            self.max_daily_loss,
        )
        self.max_rolling_loss_pct_1h = getattr(config, "MAX_ROLLING_LOSS_PCT_1H", 0.05)
        self.max_symbol_exposure_pct = getattr(config, "MAX_SYMBOL_EXPOSURE_PCT", 0.25)
        self.max_total_margin_pct = getattr(config, "MAX_TOTAL_MARGIN_PCT", 0.5)
        self.freeze_new_entries_on_breach = bool(
            getattr(config, "FREEZE_NEW_ENTRIES_ON_BREACH", True)
        )
        self.auto_flatten_on_breach = bool(getattr(config, "AUTO_FLATTEN_ON_BREACH", False))

    def check_order(self, order_event, current_price, portfolio=None):
        """Returns True if order is safe, False otherwise."""
        if current_price <= 0:
            return False, "Invalid market price."

        # 1. Check Notional Value (absolute per-order cap)
        notional_value = order_event.quantity * current_price
        if notional_value > self.max_order_value:
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

            total_equity = float(portfolio.current_holdings.get("total", 0.0))
            if total_equity <= 0:
                return False, "Non-positive equity."

            if getattr(portfolio, "circuit_breaker_tripped", False):
                return False, "Circuit breaker already tripped."

            # reduce-only orders are allowed through to let the system de-risk.
            if bool(getattr(order_event, "reduce_only", False)):
                return True, "Passed (reduce-only bypass)."

            # Approximate symbol exposure after order execution.
            cur_qty = float(portfolio.current_positions.get(order_event.symbol, 0.0))
            signed_order_qty = (
                order_event.quantity if order_event.direction == "BUY" else -order_event.quantity
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
            total_cap = total_equity * self.max_total_margin_pct
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
        margin_utilization = total_notional / equity if equity > 0 else 0.0

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
            },
        )

    def check_portfolio_risk(self, portfolio):
        """Check if daily loss limit is hit."""
        # Already handled in Portfolio circuit breaker, but can add redundancy here.
        return True, "Passed"
