from __future__ import annotations

from datetime import datetime, timedelta

from lumina_quant.backtesting.data import HistoricCSVDataHandler


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_historic_handler_accepts_prefrozen_rows_without_polars_frame():
    start = datetime(2024, 1, 1)
    rows = tuple(
        (
            start + timedelta(minutes=i),
            100.0 + i,
            101.0 + i,
            99.0 + i,
            100.5 + i,
            1000.0,
        )
        for i in range(3)
    )
    events = _Queue()
    handler = HistoricCSVDataHandler(
        events=events,
        csv_dir="data",
        symbol_list=["BTC/USDT"],
        start_date=None,
        end_date=None,
        data_dict={"BTC/USDT": rows},
    )

    assert handler.continue_backtest is True
    handler.update_bars()
    handler.update_bars()
    assert len(events.items) == 2
