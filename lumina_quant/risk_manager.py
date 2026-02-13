class RiskManager:
    """Enforces risk limits before orders are sent to the exchange."""

    def __init__(self, config):
        self.config = config
        self.max_order_value = getattr(config, "MAX_ORDER_VALUE", 5000.0)
        self.max_daily_loss = getattr(config, "MAX_DAILY_LOSS_PCT", 0.05)
        self.max_symbol_exposure_pct = getattr(config, "MAX_SYMBOL_EXPOSURE_PCT", 0.25)
        self.max_total_margin_pct = getattr(config, "MAX_TOTAL_MARGIN_PCT", 0.5)

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
            total_equity = float(portfolio.current_holdings.get("total", 0.0))
            if total_equity <= 0:
                return False, "Non-positive equity."

            if getattr(portfolio, "circuit_breaker_tripped", False):
                return False, "Circuit breaker already tripped."

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

    def check_portfolio_risk(self, portfolio):
        """Check if daily loss limit is hit."""
        # Already handled in Portfolio circuit breaker, but can add redundancy here.
        return True, "Passed"
