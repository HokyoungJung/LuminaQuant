from __future__ import annotations

import queue
import textwrap
from types import SimpleNamespace


def test_live_staleness_keys_are_runtime_loaded_via_config_module(tmp_path, monkeypatch):
    cfg = textwrap.dedent(
        """
        trading:
          symbols: ["BTC/USDT"]
        live:
          mode: "paper"
          exchange:
            driver: "binance_futures"
            name: "binance"
            market_type: "future"
            position_mode: "HEDGE"
            margin_mode: "isolated"
            leverage: 2
          materialized_staleness_threshold_seconds: 12
          materialized_staleness_alert_cooldown_seconds: 34
        """
    ).strip()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")

    monkeypatch.setenv("LQ_CONFIG_PATH", str(cfg_path))

    from lumina_quant.configuration import get_default_runtime_config
    from lumina_quant.configuration.views import LiveConfigView

    rt = get_default_runtime_config()
    assert int(rt.live.materialized_staleness_threshold_seconds) == 12
    assert int(rt.live.materialized_staleness_alert_cooldown_seconds) == 34

    live_cfg = LiveConfigView(rt)
    assert int(live_cfg.MATERIALIZED_STALENESS_THRESHOLD_SECONDS) == 12
    assert int(live_cfg.MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS) == 34

    class _ReaderStub:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        @staticmethod
        def read_snapshot():
            return SimpleNamespace(
                event_time_ms=1,
                event_time_watermark_ms=1,
                bars_1s={"BTC/USDT": tuple()},
                commit_id=None,
                lag_ms=0,
                is_stale=False,
            )

    class _ThreadStub:
        def __init__(self, target, daemon=True):
            _ = target, daemon

        @staticmethod
        def start():
            return None

    monkeypatch.setattr("lumina_quant.live.data_materialized.MaterializedWindowReader", _ReaderStub)
    monkeypatch.setattr("lumina_quant.live.data_materialized.threading.Thread", _ThreadStub)

    from lumina_quant.live.data_poll import LiveDataHandler

    handler = LiveDataHandler(
        queue.Queue(),
        ["BTC/USDT"],
        live_cfg,
        exchange=SimpleNamespace(),
    )
    assert int(handler._staleness_threshold_seconds) == 12
