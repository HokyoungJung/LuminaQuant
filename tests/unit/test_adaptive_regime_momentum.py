import queue
from dataclasses import dataclass

from lumina_quant.core.events import MarketWindowEvent
from lumina_quant.strategies.adaptive_regime_momentum import AdaptiveRegimeMomentumStrategy


@dataclass
class _WindowBarStore:
    symbol_list: list[str]

    def __post_init__(self) -> None:
        self._latest_close = dict.fromkeys(self.symbol_list)
        self._latest_time = dict.fromkeys(self.symbol_list)

    def set_bar(self, symbol: str, time_index: int, close_price: float) -> None:
        self._latest_time[symbol] = time_index
        self._latest_close[symbol] = float(close_price)

    def get_latest_bar_value(self, symbol: str, value_type: str) -> float:
        _ = value_type
        value = self._latest_close.get(symbol)
        return float(value) if value is not None else 0.0

    def get_latest_bar_datetime(self, symbol: str) -> int | None:
        return self._latest_time.get(symbol)


def _event(time_index: int, frame: dict[str, float]) -> MarketWindowEvent:
    return MarketWindowEvent(
        time=time_index,
        window_seconds=60,
        bars_1s={
            symbol: ((time_index, price, price, price, price, 1000.0),)
            for symbol, price in frame.items()
        },
    )


def _drain(events: queue.Queue) -> list:
    out = []
    while not events.empty():
        out.append(events.get())
    return out


def _feed(strategy: AdaptiveRegimeMomentumStrategy, bars: _WindowBarStore, rows):
    for time_index, frame in rows:
        for symbol, price in frame.items():
            bars.set_bar(symbol, time_index, price)
        strategy.calculate_signals_window(_event(time_index, frame), aggregator=None)
    return _drain(strategy.events)


def _risk_control_strategy(
    bars: _WindowBarStore,
    events: queue.Queue,
    **overrides,
) -> AdaptiveRegimeMomentumStrategy:
    params = {
        "lookback_bars": 4,
        "short_lookback_bars": 1,
        "regime_lookback_bars": 4,
        "volatility_lookback_bars": 10,
        "rebalance_bars": 1,
        "signal_threshold": 0.002,
        "max_longs": 1,
        "max_shorts": 0,
        "gross_exposure": 0.30,
        "stop_loss_pct": 0.0,
        "take_profit_pct": 0.0,
        "trailing_exit_pct": 0.0,
        "max_hold_bars": 0,
        "risk_off_exit": False,
    }
    params.update(overrides)
    return AdaptiveRegimeMomentumStrategy(bars, events, **params)


def _trend_rows(length: int = 80):
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    prices = {
        "BTC/USDT": 100.0,
        "ETH/USDT": 100.0,
        "BNB/USDT": 100.0,
        "SOL/USDT": 100.0,
        "TRX/USDT": 100.0,
    }
    rows = []
    for idx in range(length):
        frame = {}
        for symbol in symbols:
            step = {
                "BTC/USDT": 0.0010,
                "ETH/USDT": 0.0030,
                "BNB/USDT": 0.0020,
                "SOL/USDT": -0.0010,
                "TRX/USDT": -0.0020,
            }[symbol]
            prices[symbol] *= 1.0 + step
            frame[symbol] = prices[symbol]
        rows.append((idx, frame))
    return symbols, rows


def _regime_flip_rows(length: int = 120):
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    prices = dict.fromkeys(symbols, 100.0)
    rows = []
    for idx in range(length):
        frame = {}
        risk_off = idx >= length // 2
        for symbol in symbols:
            if not risk_off:
                step = 0.0020 if symbol in {"BTC/USDT", "ETH/USDT"} else 0.0010
            else:
                step = -0.0030 if symbol in {"BTC/USDT", "ETH/USDT", "SOL/USDT"} else -0.0010
            prices[symbol] *= 1.0 + step
            frame[symbol] = prices[symbol]
        rows.append((idx, frame))
    return symbols, rows


def test_market_window_path_generates_one_bar_per_decision_and_long_signal() -> None:
    symbols, rows = _trend_rows()
    bars = _WindowBarStore(symbols)
    events = queue.Queue()
    strategy = AdaptiveRegimeMomentumStrategy(
        bars,
        events,
        lookback_bars=8,
        short_lookback_bars=3,
        regime_lookback_bars=8,
        volatility_lookback_bars=80,
        rebalance_bars=2,
        signal_threshold=0.004,
        max_longs=1,
        max_shorts=1,
        gross_exposure=0.50,
        stop_loss_pct=0.03,
        take_profit_pct=0.0,
        trailing_exit_pct=0.012,
        max_hold_bars=0,
    )

    signals = _feed(strategy, bars, rows)

    assert len(strategy._price_history["BTC/USDT"]) == len(rows)
    long_signal = next(signal for signal in signals if signal.signal_type == "LONG")
    assert long_signal.symbol == "ETH/USDT"
    assert long_signal.metadata["target_allocation"] == 0.50
    assert strategy._quote_volume_history["BTC/USDT"]


def test_asymmetric_dynamic_risk_controls_scale_long_exposure_and_trailing() -> None:
    symbols, rows = _trend_rows()
    bars = _WindowBarStore(symbols)
    events = queue.Queue()
    strategy = AdaptiveRegimeMomentumStrategy(
        bars,
        events,
        lookback_bars=8,
        short_lookback_bars=3,
        regime_lookback_bars=8,
        volatility_lookback_bars=8,
        rebalance_bars=2,
        signal_threshold=0.004,
        max_longs=1,
        max_shorts=1,
        gross_exposure=0.50,
        long_exposure_multiplier=0.70,
        short_exposure_multiplier=0.20,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        trailing_exit_pct=0.012,
        volatility_trailing_multiplier=5.0,
        min_dynamic_trailing_pct=0.01,
        max_dynamic_trailing_pct=0.03,
        volume_weighted_broad=True,
        min_entry_volume_z=-999.0,
        max_hold_bars=0,
    )

    signals = _feed(strategy, bars, rows)

    long_signal = next(signal for signal in signals if signal.signal_type == "LONG")
    assert long_signal.metadata["target_allocation"] == 0.35
    assert 0.01 <= long_signal.metadata["effective_trailing_pct"] <= 0.03


def test_volatility_targeting_scales_entry_allocation_without_changing_signal() -> None:
    symbols, rows = _trend_rows()
    bars = _WindowBarStore(symbols)
    events = queue.Queue()
    strategy = AdaptiveRegimeMomentumStrategy(
        bars,
        events,
        lookback_bars=8,
        short_lookback_bars=3,
        regime_lookback_bars=8,
        volatility_lookback_bars=8,
        rebalance_bars=2,
        signal_threshold=0.004,
        max_longs=1,
        max_shorts=1,
        gross_exposure=0.50,
        volatility_target_per_bar=0.0005,
        min_volatility_exposure_multiplier=0.25,
        max_volatility_exposure_multiplier=0.80,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        trailing_exit_pct=0.0,
        max_hold_bars=0,
    )

    signals = _feed(strategy, bars, rows)

    long_signal = next(signal for signal in signals if signal.signal_type == "LONG")
    assert long_signal.symbol == "ETH/USDT"
    assert 0.125 <= long_signal.metadata["target_allocation"] <= 0.40
    assert long_signal.metadata["volatility_exposure_multiplier"] <= 0.80


def test_regime_flip_can_exit_longs_and_enter_short() -> None:
    symbols, rows = _regime_flip_rows()
    bars = _WindowBarStore(symbols)
    events = queue.Queue()
    strategy = AdaptiveRegimeMomentumStrategy(
        bars,
        events,
        lookback_bars=8,
        short_lookback_bars=3,
        regime_lookback_bars=8,
        rebalance_bars=2,
        signal_threshold=0.004,
        max_longs=1,
        max_shorts=1,
        gross_exposure=0.40,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        trailing_exit_pct=0.0,
        max_hold_bars=0,
    )

    signals = _feed(strategy, bars, rows)
    signal_types = [signal.signal_type for signal in signals]

    assert "LONG" in signal_types
    assert "EXIT" in signal_types
    assert "SHORT" in signal_types


def test_state_roundtrip_preserves_signal_sequence() -> None:
    symbols, rows = _regime_flip_rows()
    params = {
        "lookback_bars": 8,
        "short_lookback_bars": 3,
        "regime_lookback_bars": 8,
        "rebalance_bars": 2,
        "signal_threshold": 0.004,
        "max_longs": 1,
        "max_shorts": 1,
        "gross_exposure": 0.40,
        "stop_loss_pct": 0.0,
        "take_profit_pct": 0.0,
        "trailing_exit_pct": 0.0,
        "max_hold_bars": 0,
    }

    full_bars = _WindowBarStore(symbols)
    full_events = queue.Queue()
    full_strategy = AdaptiveRegimeMomentumStrategy(full_bars, full_events, **params)
    full_signals = [
        (signal.datetime, signal.symbol, signal.signal_type)
        for signal in _feed(full_strategy, full_bars, rows)
    ]

    split = 70
    bars_a = _WindowBarStore(symbols)
    events_a = queue.Queue()
    strategy_a = AdaptiveRegimeMomentumStrategy(bars_a, events_a, **params)
    signals_a = _feed(strategy_a, bars_a, rows[:split])
    saved_state = strategy_a.get_state()

    bars_b = _WindowBarStore(symbols)
    events_b = queue.Queue()
    strategy_b = AdaptiveRegimeMomentumStrategy(bars_b, events_b, **params)
    strategy_b.set_state(saved_state)
    signals_b = _feed(strategy_b, bars_b, rows[split:])

    split_signals = [
        (signal.datetime, signal.symbol, signal.signal_type) for signal in [*signals_a, *signals_b]
    ]
    assert split_signals == full_signals
