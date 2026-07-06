import os
import unittest

from lumina_quant.configuration import validate_runtime_config
from lumina_quant.configuration.schema import RuntimeConfig


class TestConfigSchema(unittest.TestCase):
    def test_exchange_schema_exists(self):
        rt = RuntimeConfig()
        # LiveExchangeConfig dataclass has these typed fields
        self.assertTrue(hasattr(rt.live.exchange, "driver"))
        self.assertTrue(hasattr(rt.live.exchange, "name"))
        self.assertTrue(hasattr(rt.live.exchange, "market_type"))

    def test_real_mode_requires_explicit_flag(self):
        rt = RuntimeConfig()
        rt.live.mode = "real"
        rt.live.require_real_enable_flag = True
        rt.live.api_key = "test_key"
        rt.live.secret_key = "test_secret"
        # Real-money live-safety gate (real_money_readiness audit 2026-07-06) fail-closed
        # requirements: pin explicit symbols (M6) and arm the live-safety knobs (C1-C4/C6)
        # so this test isolates the LUMINA_ENABLE_LIVE_REAL env flag, not the safety gate.
        rt.trading.symbols = ["BTC/USDT"]
        rt.risk.auto_flatten_on_breach = True
        rt.risk.flatten_escalate_to_market = True
        rt.live.real_mode_managed_protective_stops = True
        rt.live.max_limit_price_band_pct = 0.02
        rt.live.equity_reconciliation_interval_sec = 30.0
        rt.live.market_data_silence_timeout_sec = 45.0
        rt.live.bar_sanity_check = True
        rt.live.stale_symbol_after_ms = 5_000

        old_env = os.environ.get("LUMINA_ENABLE_LIVE_REAL")
        try:
            os.environ.pop("LUMINA_ENABLE_LIVE_REAL", None)
            with self.assertRaises(ValueError):
                validate_runtime_config(rt, for_live=True)

            os.environ["LUMINA_ENABLE_LIVE_REAL"] = "true"
            validate_runtime_config(rt, for_live=True)
        finally:
            if old_env is None:
                os.environ.pop("LUMINA_ENABLE_LIVE_REAL", None)
            else:
                os.environ["LUMINA_ENABLE_LIVE_REAL"] = old_env


if __name__ == "__main__":
    unittest.main()
