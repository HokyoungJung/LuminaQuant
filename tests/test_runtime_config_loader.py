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

    def test_promotion_gate_strategy_profile_loaded(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            promotion_gate:
              days: 14
              max_order_rejects: 0
              strategy_profiles:
                RsiStrategy:
                  days: 21
                  max_order_rejects: 2
                  require_alpha_card: true
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
            runtime = load_runtime_config(config_path=path, env=os.environ)
            self.assertEqual(runtime.promotion_gate.days, 14)
            profile = runtime.promotion_gate.strategy_profiles["RsiStrategy"]
            self.assertEqual(int(profile["days"]), 21)
            self.assertEqual(int(profile["max_order_rejects"]), 2)
            self.assertTrue(bool(profile["require_alpha_card"]))
        finally:
            os.remove(path)

    def test_env_overrides_apply_backtest_and_gpu_fields(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            execution:
              compute_backend: "auto"
            backtest:
              chunk_days: 7
              skip_ahead_enabled: true
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
            env["LQ__BACKTEST__CHUNK_DAYS"] = "2"
            env["LQ__BACKTEST__CHUNK_WARMUP_BARS"] = "11"
            env["LQ__BACKTEST__SKIP_AHEAD_ENABLED"] = "0"
            env["LQ__BACKTEST__DECISION_CADENCE_SECONDS"] = "15"
            env["LQ__BACKTEST__POLL_SECONDS"] = "21"
            env["LQ__BACKTEST__WINDOW_SECONDS"] = "22"
            env["LQ__EXECUTION__GPU_MODE"] = "gpu"
            env["LQ__EXECUTION__GPU_VRAM_GB"] = "6.5"
            env["LQ__LIVE__POLL_SECONDS"] = "4"
            env["LQ__LIVE__WINDOW_SECONDS"] = "5"
            runtime = load_runtime_config(config_path=path, env=env)
            self.assertEqual(runtime.backtest.chunk_days, 2)
            self.assertEqual(runtime.backtest.chunk_warmup_bars, 11)
            self.assertFalse(runtime.backtest.skip_ahead_enabled)
            self.assertEqual(runtime.backtest.decision_cadence_seconds, 15)
            self.assertEqual(runtime.backtest.backtest_decision_seconds, 15)
            self.assertEqual(runtime.backtest.poll_seconds, 21)
            self.assertEqual(runtime.backtest.backtest_poll_seconds, 21)
            self.assertEqual(runtime.backtest.window_seconds, 22)
            self.assertEqual(runtime.backtest.backtest_window_seconds, 22)
            self.assertEqual(runtime.execution.gpu_mode, "gpu")
            self.assertEqual(runtime.execution.compute_backend, "gpu")
            self.assertAlmostEqual(runtime.execution.gpu_vram_gb, 6.5)
            self.assertEqual(runtime.live.poll_seconds, 4)
            self.assertEqual(runtime.live.poll_interval, 4)
            self.assertEqual(runtime.live.live_poll_seconds, 4)
            self.assertEqual(runtime.live.window_seconds, 5)
            self.assertEqual(runtime.live.ingest_window_seconds, 5)
        finally:
            os.remove(path)

    def test_env_backtest_decision_cadence_seconds(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            backtest:
              decision_cadence_seconds: 20
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
            env["LQ__BACKTEST__DECISION_CADENCE_SECONDS"] = "33"
            runtime = load_runtime_config(config_path=path, env=env)
            self.assertEqual(runtime.backtest.decision_cadence_seconds, 33)
            self.assertEqual(runtime.backtest.backtest_decision_seconds, 33)
        finally:
            os.remove(path)

    def test_trading_timeframes_and_recent_split_days(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
              timeframe: "1D"
              timeframes: ["1m", "4H", "1D"]
            optimization:
              validation_days: 30
              oos_days: 30
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
            runtime = load_runtime_config(config_path=path, env=os.environ)
            self.assertEqual(runtime.trading.timeframe, "1d")
            self.assertEqual(runtime.trading.timeframes, ["1m", "4h", "1d"])
            self.assertEqual(runtime.optimization.validation_days, 30)
            self.assertEqual(runtime.optimization.oos_days, 30)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
