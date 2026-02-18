"""Tests for typed runtime config loader."""

import os
import tempfile
import textwrap
import unittest

from lumina_quant.configuration.loader import load_runtime_config


class TestRuntimeConfigLoader(unittest.TestCase):
    """Runtime config loader coverage for env overrides and strict settings."""

    def test_env_nested_override(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            live:
              mode: "paper"
              exchange:
                driver: "ccxt"
                name: "binance"
                market_type: "future"
                position_mode: "HEDGE"
                margin_mode: "isolated"
                leverage: 2
            """
        ).strip()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fp:
            fp.write(yaml_text)
            path = fp.name
        try:
            env = dict(os.environ)
            env["LQ__LIVE__EXCHANGE__LEVERAGE"] = "3"
            runtime = load_runtime_config(config_path=path, env=env)
            self.assertEqual(runtime.live.exchange.leverage, 3)
        finally:
            os.remove(path)

    def test_live_mode_is_explicit(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            live:
              mode: "real"
              exchange:
                driver: "ccxt"
                name: "binance"
                market_type: "future"
                position_mode: "HEDGE"
                margin_mode: "isolated"
                leverage: 2
            """
        ).strip()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fp:
            fp.write(yaml_text)
            path = fp.name
        try:
            runtime = load_runtime_config(config_path=path, env=os.environ)
            self.assertEqual(runtime.live.mode, "real")
        finally:
            os.remove(path)

    def test_invalid_compute_backend_raises(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            execution:
              compute_backend: "torch"
            live:
              mode: "paper"
              exchange:
                driver: "ccxt"
                name: "binance"
                market_type: "future"
                position_mode: "HEDGE"
                margin_mode: "isolated"
                leverage: 2
            """
        ).strip()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fp:
            fp.write(yaml_text)
            path = fp.name
        try:
            with self.assertRaises(ValueError):
                load_runtime_config(config_path=path, env=os.environ)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
