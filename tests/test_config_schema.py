import os
import unittest

from lumina_quant.config import LiveConfig


class TestConfigSchema(unittest.TestCase):
    def test_exchange_schema_exists(self):
        self.assertIsInstance(LiveConfig.EXCHANGE, dict)
        self.assertIn("driver", LiveConfig.EXCHANGE)
        self.assertIn("name", LiveConfig.EXCHANGE)
        self.assertIn("market_type", LiveConfig.EXCHANGE)

    def test_real_mode_requires_explicit_flag(self):
        original = {
            "MODE": LiveConfig.MODE,
            "REQUIRE_REAL_ENABLE_FLAG": LiveConfig.REQUIRE_REAL_ENABLE_FLAG,
            "BINANCE_API_KEY": LiveConfig.BINANCE_API_KEY,
            "BINANCE_SECRET_KEY": LiveConfig.BINANCE_SECRET_KEY,
        }
        old_env = os.environ.get("LUMINA_ENABLE_LIVE_REAL")
        try:
            LiveConfig.MODE = "real"
            LiveConfig.REQUIRE_REAL_ENABLE_FLAG = True
            LiveConfig.BINANCE_API_KEY = "test_key"
            LiveConfig.BINANCE_SECRET_KEY = "test_secret"

            if "LUMINA_ENABLE_LIVE_REAL" in os.environ:
                del os.environ["LUMINA_ENABLE_LIVE_REAL"]
            with self.assertRaises(ValueError):
                LiveConfig.validate()

            os.environ["LUMINA_ENABLE_LIVE_REAL"] = "true"
            LiveConfig.validate()
        finally:
            LiveConfig.MODE = original["MODE"]
            LiveConfig.REQUIRE_REAL_ENABLE_FLAG = original["REQUIRE_REAL_ENABLE_FLAG"]
            LiveConfig.BINANCE_API_KEY = original["BINANCE_API_KEY"]
            LiveConfig.BINANCE_SECRET_KEY = original["BINANCE_SECRET_KEY"]
            if old_env is None:
                os.environ.pop("LUMINA_ENABLE_LIVE_REAL", None)
            else:
                os.environ["LUMINA_ENABLE_LIVE_REAL"] = old_env


if __name__ == "__main__":
    unittest.main()
