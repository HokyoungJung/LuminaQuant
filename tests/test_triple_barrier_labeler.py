from __future__ import annotations

from lumina_quant.research.triple_barrier import BarrierType, label_triple_barrier


def test_take_profit_label_uses_first_profit_bar_after_costs() -> None:
    outcome = label_triple_barrier(
        [100.0, 101.0, 104.0, 103.0],
        entry_index=0,
        side="LONG",
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        max_hold_bars=3,
        fee_bps=1.0,
        slippage_bps=2.0,
    )
    assert outcome.barrier_type == BarrierType.TAKE_PROFIT
    assert outcome.exit_index == 2
    assert outcome.gross_pnl > 0.03
    assert outcome.net_pnl_bps < outcome.gross_pnl * 10_000.0


def test_stop_loss_is_conservative_when_intrabar_stop_and_take_overlap() -> None:
    outcome = label_triple_barrier(
        [100.0, 101.0, 103.0],
        highs=[100.0, 105.0, 103.0],
        lows=[100.0, 96.0, 102.0],
        entry_index=0,
        side="LONG",
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        max_hold_bars=2,
    )
    assert outcome.barrier_type == BarrierType.STOP_LOSS
    assert outcome.exit_index == 1
    assert outcome.max_adverse_excursion <= -0.02
    assert outcome.max_favorable_excursion >= 0.03


def test_short_time_exit_reports_mae_mfe_and_funding_cost() -> None:
    outcome = label_triple_barrier(
        [100.0, 99.0, 98.0, 99.0],
        entry_index=0,
        side="SHORT",
        stop_loss_pct=0.05,
        take_profit_pct=0.05,
        max_hold_bars=3,
        funding_bps_per_bar=1.0,
    )
    assert outcome.barrier_type == BarrierType.TIME_EXIT
    assert outcome.side == "SHORT"
    assert outcome.bars_held == 3
    assert outcome.funding_cost == 0.0003
    assert outcome.net_pnl_bps > 0.0
