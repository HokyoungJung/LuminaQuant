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
