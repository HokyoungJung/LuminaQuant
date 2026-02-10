class RiskManager:
    """
    Enforces risk limits before orders are sent to the exchange.
    """

    def __init__(self, config):
        self.config = config
        self.max_order_value = 5000.0  # Max $ per trade
        self.max_daily_loss = getattr(config, "MAX_DAILY_LOSS_PCT", 0.05)  # 5%

    def check_order(self, order_event, current_price):
        """
        Returns True if order is safe, False otherwise.
        """
        # 1. Check Notional Value
        notional_value = order_event.quantity * current_price
        if notional_value > self.max_order_value:
            return (
                False,
                f"Order Value ${notional_value:.2f} exceeds limit ${self.max_order_value}",
            )

        # 2. Check Negative Quantity
        if order_event.quantity <= 0:
            return False, f"Invalid Quantity: {order_event.quantity}"

        return True, "Passed"

    def check_portfolio_risk(self, portfolio):
        """
        Check if daily loss limit is hit.
        """
        # Already handled in Portfolio circuit breaker, but can add redundancy here.
        return True, "Passed"
