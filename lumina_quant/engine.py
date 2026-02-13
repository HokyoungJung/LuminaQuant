from abc import ABC


class TradingEngine(ABC):
    """Abstract base class for trading engines (Backtest and LiveTrader).
    Encapsulates common event processing logic (The "Kernel").
    """

    def __init__(self, events, data_handler, strategy, portfolio, execution_handler):
        self.events = events
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler

        # Stats
        self.market_events = 0
        self.signals = 0
        self.orders = 0
        self.fills = 0

    def process_event(self, event):
        """Routing logic for events."""
        if event is not None:
            if event.type == "MARKET":
                self.handle_market_event(event)
            elif event.type == "SIGNAL":
                self.handle_signal_event(event)
            elif event.type == "ORDER":
                self.handle_order_event(event)
            elif event.type == "FILL":
                self.handle_fill_event(event)

    def handle_market_event(self, event):
        self.market_events += 1
        self.strategy.calculate_signals(event)
        self.portfolio.update_timeindex(event)
        # Optional: Simulated execution handler might need to check open orders
        if hasattr(self.execution_handler, "check_open_orders"):
            self.execution_handler.check_open_orders(event)

    def handle_signal_event(self, event):
        self.signals += 1
        self.portfolio.update_signal(event)

    def handle_order_event(self, event):
        self.orders += 1
        self.execution_handler.execute_order(event)

    def handle_fill_event(self, event):
        self.fills += 1
        self.portfolio.update_fill(event)
        # Hook for state saving (LiveTrader can override or we add a hook)
        self.on_fill(event)

    def on_fill(self, event):
        """Hook for post-fill actions (e.g. logging, saving state).
        Override in subclasses.
        """
        pass
