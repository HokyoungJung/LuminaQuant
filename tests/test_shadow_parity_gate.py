"""Phase 5 shadow parity gate — ShadowLiveRunner expansion tests.

Tests cover:
- Backward compat: existing run() signature with new parity_ratio field
- evaluate_signal_parity: identical / diverging / error-raising strategy fns
- parity_ratio gate: strategy-fn-absent → perfect parity (gate passes)
- window truncation at parity_window_bars
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from lumina_quant.live.shadow_live_runner import ShadowLiveResult, ShadowLiveRunner


# ── ShadowLiveResult backward compat ────────────────────────────────────────


def test_shadow_result_has_parity_ratio_field():
    r = ShadowLiveResult(events_processed=10, divergence_count=0)
    assert hasattr(r, "parity_ratio")
    assert r.parity_ratio == 1.0  # default


def test_shadow_result_has_signal_fields():
    r = ShadowLiveResult(events_processed=5, divergence_count=1)
    assert r.signal_matches == 0
    assert r.signal_total == 0


# ── run() backward compat ────────────────────────────────────────────────────


def test_run_zero_divergence_identical_events():
    events = [
        SimpleNamespace(timestamp_ns=1, sequence=1),
        SimpleNamespace(timestamp_ns=2, sequence=2),
    ]
    runner = ShadowLiveRunner()
    result = runner.run(baseline_events=events, candidate_events=list(events))
    assert result.events_processed == 2
    assert result.divergence_count == 0
    assert result.parity_ratio == 1.0


def test_run_sequence_mismatch_reports_divergence():
    baseline = [SimpleNamespace(timestamp_ns=1, sequence=1)]
    candidate = [SimpleNamespace(timestamp_ns=1, sequence=2)]
    runner = ShadowLiveRunner()
    result = runner.run(baseline_events=baseline, candidate_events=candidate)
    assert result.events_processed == 1
    assert result.divergence_count == 1
    assert result.parity_ratio == 0.0


def test_run_count_delta_counts_as_divergence():
    events = [
        SimpleNamespace(timestamp_ns=1, sequence=1),
        SimpleNamespace(timestamp_ns=2, sequence=2),
    ]
    runner = ShadowLiveRunner()
    result = runner.run(baseline_events=events, candidate_events=events[:1])
    assert result.divergence_count >= 1


def test_run_empty_events_returns_perfect_parity():
    runner = ShadowLiveRunner()
    result = runner.run(baseline_events=[], candidate_events=[])
    assert result.parity_ratio == 1.0
    assert result.events_processed == 0


# ── evaluate_signal_parity ───────────────────────────────────────────────────


def test_evaluate_parity_identical_strategy_fns():
    """Identical strategy functions → parity_ratio == 1.0."""
    windows = [SimpleNamespace(close=float(i)) for i in range(10)]
    fn = lambda w: w.close * 1.5  # noqa: E731
    runner = ShadowLiveRunner(baseline_strategy_fn=fn, candidate_strategy_fn=fn)
    result = runner.evaluate_signal_parity(market_windows=windows)
    assert result.parity_ratio == 1.0
    assert result.signal_matches == 10
    assert result.signal_total == 10
    assert result.divergence_count == 0


def test_evaluate_parity_diverging_strategy_fns():
    """Always-different strategy functions → parity_ratio == 0.0."""
    windows = [SimpleNamespace(close=1.0)] * 5
    runner = ShadowLiveRunner(
        baseline_strategy_fn=lambda w: 1.0,
        candidate_strategy_fn=lambda w: -1.0,
    )
    result = runner.evaluate_signal_parity(market_windows=windows)
    assert result.parity_ratio == 0.0
    assert result.signal_matches == 0
    assert result.signal_total == 5


def test_evaluate_parity_partial_agreement():
    """Half-matching strategy → parity_ratio == 0.5."""
    toggle = {"v": True}

    def candidate(w):
        toggle["v"] = not toggle["v"]
        return 1.0 if toggle["v"] else -1.0

    windows = [SimpleNamespace()] * 4
    runner = ShadowLiveRunner(
        baseline_strategy_fn=lambda w: 1.0,
        candidate_strategy_fn=candidate,
    )
    result = runner.evaluate_signal_parity(market_windows=windows)
    assert result.signal_total == 4
    assert 0.0 < result.parity_ratio < 1.0


def test_evaluate_parity_no_strategy_fns_returns_perfect_parity():
    """No strategy functions configured → gate does not block (parity_ratio=1.0)."""
    runner = ShadowLiveRunner()
    windows = [SimpleNamespace()] * 5
    result = runner.evaluate_signal_parity(market_windows=windows)
    assert result.parity_ratio == 1.0
    assert result.signal_total == 0  # no comparisons made
    assert result.events_processed == 5


def test_evaluate_parity_strategy_error_counts_as_divergence():
    """Strategy functions that raise count the bar as a divergence."""

    def bad_fn(w):
        raise RuntimeError("explode")

    runner = ShadowLiveRunner(
        baseline_strategy_fn=bad_fn,
        candidate_strategy_fn=lambda w: 0.0,
    )
    windows = [SimpleNamespace()] * 3
    result = runner.evaluate_signal_parity(market_windows=windows)
    assert result.signal_total == 3
    assert result.signal_matches == 0
    assert result.parity_ratio == 0.0


def test_evaluate_parity_nan_signal_does_not_match():
    """NaN signals are not finite and thus never match."""
    runner = ShadowLiveRunner(
        baseline_strategy_fn=lambda w: float("nan"),
        candidate_strategy_fn=lambda w: float("nan"),
    )
    windows = [SimpleNamespace()] * 2
    result = runner.evaluate_signal_parity(market_windows=windows)
    assert result.signal_matches == 0
    assert result.parity_ratio == 0.0


# ── Fill / cost parity: live paper vs backtest ExecutionModel ────────────────


def test_live_paper_lmt_fill_commission_matches_execution_model():
    """LMT fill commission from live paper path must match backtest ExecutionModel.

    This validates the Phase 5 requirement that backtest ↔ live share a single
    cost formula.  LMT fills are deterministic (no random slippage) so the
    comparison is exact within floating-point tolerance.
    """
    import queue

    from lumina_quant.backtesting.execution_model import ExecutionModel, ExecutionModelConfig
    from lumina_quant.live.execution_live import LiveExecutionHandler

    FILL_PRICE = 50_000.0
    FILL_QTY = 0.01
    MAKER_FEE = 0.0002

    events_q = queue.Queue()
    bars = SimpleNamespace(
        get_latest_bar_value=lambda *a, **kw: FILL_PRICE,
        get_latest_bar_datetime=lambda *a, **kw: None,
    )
    config = SimpleNamespace(
        MODE="paper",
        ORDER_TIMEOUT=10,
        ORDER_STATE_SOURCE="polling",
        RECONCILIATION_POLL_FALLBACK_ENABLED=True,
        TAKER_FEE_RATE=0.0004,
        MAKER_FEE_RATE=MAKER_FEE,
        SLIPPAGE_RATE=0.0005,
        SPREAD_RATE=0.0002,
        LEVERAGE=3,
        MARGIN_MODE="isolated",
        MAINTENANCE_MARGIN_RATE=0.005,
        LIQUIDATION_BUFFER_RATE=0.0005,
        FUNDING_RATE_PER_8H=0.0,
        FUNDING_INTERVAL_HOURS=8,
        MARKET_TYPE="future",
        PAPER_EXCHANGE_PROTECTIVE_ORDERS=False,
        ALLOW_MARKET_ORDERS=False,
        EXCHANGE_ID="TESTNET",
    )

    class _FakeExchange:
        pass

    handler = LiveExecutionHandler(events_q, bars, config, _FakeExchange())

    event = SimpleNamespace(
        type="ORDER",
        symbol="BTC/USDT",
        direction="BUY",
        quantity=FILL_QTY,
        order_type="LMT",
        price=FILL_PRICE,
        position_side="LONG",
        reduce_only=False,
        client_order_id="test-parity-lmt",
        metadata={},
    )
    order = {
        "id": "order-parity-1",
        "average": FILL_PRICE,
        "price": FILL_PRICE,
        "amount": FILL_QTY,
        "filled": FILL_QTY,
    }
    handler._emit_fill_event(event, order, FILL_QTY, status="filled")

    fill_ev = events_q.get_nowait()
    live_commission = fill_ev.commission

    # Backtest reference: same config, same inputs
    em_cfg = ExecutionModelConfig(
        taker_fee_rate=0.0004,
        maker_fee_rate=MAKER_FEE,
        slippage_rate=0.0005,
        spread_rate=0.0002,
        leverage=3,
        margin_mode="isolated",
        maintenance_margin_rate=0.005,
        liquidation_buffer_rate=0.0005,
        funding_rate_per_8h=0.0,
        funding_interval_hours=8,
        random_seed=42,
    )
    em = ExecutionModel(em_cfg)
    bt_result = em.compute_fill(
        raw_price=FILL_PRICE,
        qty=FILL_QTY,
        direction="BUY",
        bar_volume=0.0,
        is_maker=True,  # LMT → maker
        apply_liquidity_cap=False,
    )

    assert live_commission == pytest.approx(bt_result.commission, rel=1e-9), (
        f"Live paper LMT commission {live_commission!r} != "
        f"backtest ExecutionModel commission {bt_result.commission!r}"
    )
    # Sanity: LMT commission = fill_price * qty * maker_fee_rate
    assert live_commission == pytest.approx(FILL_PRICE * FILL_QTY * MAKER_FEE, rel=1e-9)


def test_live_paper_mkt_fill_commission_uses_taker_fee():
    """MKT fill in paper mode must apply taker fee (not maker fee)."""
    import queue

    from lumina_quant.live.execution_live import LiveExecutionHandler

    FILL_PRICE = 30_000.0
    FILL_QTY = 0.005
    TAKER_FEE = 0.0004

    events_q = queue.Queue()
    bars = SimpleNamespace(
        get_latest_bar_value=lambda *a, **kw: FILL_PRICE,
        get_latest_bar_datetime=lambda *a, **kw: None,
    )
    config = SimpleNamespace(
        MODE="paper",
        ORDER_TIMEOUT=10,
        ORDER_STATE_SOURCE="polling",
        RECONCILIATION_POLL_FALLBACK_ENABLED=True,
        TAKER_FEE_RATE=TAKER_FEE,
        MAKER_FEE_RATE=0.0002,
        SLIPPAGE_RATE=0.0005,
        SPREAD_RATE=0.0002,
        LEVERAGE=3,
        MARGIN_MODE="isolated",
        MAINTENANCE_MARGIN_RATE=0.005,
        LIQUIDATION_BUFFER_RATE=0.0005,
        FUNDING_RATE_PER_8H=0.0,
        FUNDING_INTERVAL_HOURS=8,
        MARKET_TYPE="future",
        PAPER_EXCHANGE_PROTECTIVE_ORDERS=False,
        ALLOW_MARKET_ORDERS=True,
        EXCHANGE_ID="TESTNET",
    )

    class _FakeExchange:
        pass

    handler = LiveExecutionHandler(events_q, bars, config, _FakeExchange())

    event = SimpleNamespace(
        type="ORDER",
        symbol="ETH/USDT",
        direction="SELL",
        quantity=FILL_QTY,
        order_type="MKT",
        price=None,
        position_side="SHORT",
        reduce_only=False,
        client_order_id="test-parity-mkt",
        metadata={},
    )
    order = {
        "id": "order-parity-2",
        "average": FILL_PRICE,
        "price": FILL_PRICE,
        "amount": FILL_QTY,
        "filled": FILL_QTY,
    }
    handler._emit_fill_event(event, order, FILL_QTY, status="filled")

    fill_ev = events_q.get_nowait()
    live_commission = fill_ev.commission

    # MKT applies slippage via compute_fill — commission is based on slipped price.
    # At minimum, commission must exceed zero and must NOT equal maker-fee * notional.
    maker_commission = FILL_PRICE * FILL_QTY * 0.0002
    assert live_commission > 0.0, "MKT commission must be positive"
    assert live_commission != pytest.approx(maker_commission, rel=1e-6), (
        "MKT fill must use taker fee path, not maker fee"
    )


def test_evaluate_parity_window_truncated_to_parity_window_bars():
    """Only up to parity_window_bars windows are evaluated."""
    runner = ShadowLiveRunner(
        baseline_strategy_fn=lambda w: 1.0,
        candidate_strategy_fn=lambda w: 1.0,
        parity_window_bars=3,
    )
    windows = [SimpleNamespace()] * 10
    result = runner.evaluate_signal_parity(market_windows=windows)
    assert result.signal_total == 3
    assert result.events_processed == 3


def test_evaluate_parity_empty_windows_returns_perfect_parity():
    runner = ShadowLiveRunner(
        baseline_strategy_fn=lambda w: 1.0,
        candidate_strategy_fn=lambda w: 1.0,
    )
    result = runner.evaluate_signal_parity(market_windows=[])
    assert result.parity_ratio == 1.0
    assert result.signal_total == 0


# ── parity gate threshold check ──────────────────────────────────────────────


def test_parity_ratio_meets_threshold():
    """Confirms caller logic: parity_ratio >= threshold → gate passes."""
    runner = ShadowLiveRunner(
        baseline_strategy_fn=lambda w: 1.0,
        candidate_strategy_fn=lambda w: 1.0,
        parity_window_bars=100,
    )
    windows = [SimpleNamespace()] * 100
    result = runner.evaluate_signal_parity(market_windows=windows)
    min_ratio = 0.99
    assert result.parity_ratio >= min_ratio, (
        f"Parity {result.parity_ratio:.4f} below threshold {min_ratio}"
    )


def test_parity_ratio_below_threshold_does_not_meet_gate():
    runner = ShadowLiveRunner(
        baseline_strategy_fn=lambda w: 1.0,
        candidate_strategy_fn=lambda w: -1.0,
        parity_window_bars=100,
    )
    windows = [SimpleNamespace()] * 100
    result = runner.evaluate_signal_parity(market_windows=windows)
    min_ratio = 0.99
    assert result.parity_ratio < min_ratio
