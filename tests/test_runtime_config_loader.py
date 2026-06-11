"""Tests for typed runtime config loader."""

import os
import tempfile
import textwrap
import unittest

from lumina_quant.configuration.loader import build_runtime_config, load_runtime_config
from lumina_quant.configuration.validate import validate_runtime_config


class TestRuntimeConfigLoader(unittest.TestCase):
    """Runtime config loader coverage for env overrides and strict settings."""

    def test_env_nested_override(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            execution:
              gpu_mode: "gpu"
              compute_backend: "gpu"
            live:
              mode: "paper"
              exchange:
                driver: "binance_futures"
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
                driver: "binance_futures"
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

    def test_live_api_credentials_are_not_loaded_from_yaml(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            live:
              mode: "paper"
              api_key: "yaml-api-key"
              secret_key: "yaml-secret-key"
              exchange:
                driver: "binance_futures"
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
            runtime = load_runtime_config(config_path=path, env={})
            self.assertEqual(runtime.live.api_key, "")
            self.assertEqual(runtime.live.secret_key, "")
        finally:
            os.remove(path)

    def test_live_api_credentials_are_loaded_from_env(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            live:
              mode: "paper"
              api_key: "yaml-api-key"
              secret_key: "yaml-secret-key"
              exchange:
                driver: "binance_futures"
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
            runtime = load_runtime_config(
                config_path=path,
                env={"BINANCE_API_KEY": "env-api-key", "EXCHANGE_SECRET_KEY": "env-secret-key"},
            )
            self.assertEqual(runtime.live.api_key, "env-api-key")
            self.assertEqual(runtime.live.secret_key, "env-secret-key")
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
                driver: "binance_futures"
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
                driver: "binance_futures"
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
                driver: "binance_futures"
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
            env["LQ__BACKTEST__MARGIN_MODE"] = "isolated"
            env["LQ__EXECUTION__GPU_MODE"] = "gpu"
            env["LQ__EXECUTION__GPU_VRAM_GB"] = "6.5"
            env["LQ__STORAGE__COLLECTOR_BOOTSTRAP_LOOKBACK_HOURS"] = "36"
            env["LQ__LIVE__POLL_SECONDS"] = "4"
            env["LQ__LIVE__WINDOW_SECONDS"] = "5"
            runtime = load_runtime_config(config_path=path, env=env)
            self.assertEqual(runtime.backtest.chunk_days, 2)
            self.assertEqual(runtime.backtest.chunk_warmup_bars, 11)
            self.assertFalse(runtime.backtest.skip_ahead_enabled)
            self.assertEqual(runtime.backtest.decision_cadence_seconds, 15)
            self.assertEqual(runtime.backtest.backtest_decision_seconds, 15)
            self.assertEqual(runtime.backtest.margin_mode, "isolated")
            self.assertEqual(runtime.backtest.poll_seconds, 21)
            self.assertEqual(runtime.backtest.backtest_poll_seconds, 21)
            self.assertEqual(runtime.backtest.window_seconds, 22)
            self.assertEqual(runtime.backtest.backtest_window_seconds, 22)
            self.assertEqual(runtime.execution.gpu_mode, "gpu")
            self.assertEqual(runtime.execution.compute_backend, "gpu")
            self.assertAlmostEqual(runtime.execution.gpu_vram_gb, 6.5)
            self.assertEqual(runtime.storage.collector_bootstrap_lookback_hours, 36)
            self.assertEqual(runtime.live.poll_seconds, 4)
            self.assertEqual(runtime.live.poll_interval, 4)
            self.assertEqual(runtime.live.live_poll_seconds, 4)
            self.assertEqual(runtime.live.window_seconds, 5)
            self.assertEqual(runtime.live.ingest_window_seconds, 5)
        finally:
            os.remove(path)

    def test_live_order_policy_defaults_limit_first(self):
        yaml_text = textwrap.dedent(
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
            """
        ).strip()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fp:
            fp.write(yaml_text)
            path = fp.name
        try:
            env = {key: value for key, value in os.environ.items() if not key.startswith("LQ__")}
            runtime = load_runtime_config(config_path=path, env=env)
            validate_runtime_config(runtime)
            self.assertEqual(runtime.live.default_order_type, "LMT")
            self.assertFalse(runtime.live.allow_market_orders)
            self.assertEqual(runtime.live.limit_price_mode, "one_tick_worse")
            self.assertEqual(runtime.live.limit_price_offset_ticks, 1)
            self.assertEqual(runtime.live.limit_time_in_force, "GTC")
            self.assertEqual(runtime.live.protective_order_style, "limit")
        finally:
            os.remove(path)

    def test_market_default_requires_explicit_live_opt_in(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            live:
              mode: "paper"
              default_order_type: "MKT"
              allow_market_orders: false
              exchange:
                driver: "binance_futures"
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
            env = {key: value for key, value in os.environ.items() if not key.startswith("LQ__")}
            runtime = load_runtime_config(config_path=path, env=env)
            with self.assertRaisesRegex(ValueError, "requires live.allow_market_orders=true"):
                validate_runtime_config(runtime)
        finally:
            os.remove(path)

    def test_execution_defaults_are_gpu_first(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            execution:
              gpu_mode: "gpu"
              compute_backend: "gpu"
            live:
              mode: "paper"
              exchange:
                driver: "binance_futures"
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
            runtime = build_runtime_config(
                {
                    "trading": {"symbols": ["BTC/USDT"]},
                    "execution": {"gpu_mode": "gpu", "compute_backend": "gpu"},
                    "live": {
                        "mode": "paper",
                        "exchange": {
                            "driver": "binance_futures",
                            "name": "binance",
                            "market_type": "future",
                            "position_mode": "HEDGE",
                            "margin_mode": "isolated",
                            "leverage": 2,
                        },
                    },
                },
                env={},
            )
            self.assertEqual(runtime.execution.gpu_mode, "gpu")
            self.assertEqual(runtime.execution.compute_backend, "gpu")
        finally:
            os.remove(path)

    def test_backtest_margin_mode_rejects_cross_until_cross_liquidation_model_exists(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            backtest:
              margin_mode: "cross"
            live:
              mode: "paper"
              exchange:
                driver: "binance_futures"
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
            with self.assertRaisesRegex(ValueError, "backtest.margin_mode"):
                validate_runtime_config(runtime)
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
                driver: "binance_futures"
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

    def test_backtest_external_config_loaded(self):
        yaml_text = textwrap.dedent(
            """
            trading:
              symbols: ["BTC/USDT"]
            backtest:
              data_source: "external"
              external:
                source_kind: "csv"
                root_path: "var/data/external/backtest"
                symbol_map:
                  BTC/USDT: "BTCUSDT.csv"
            live:
              mode: "paper"
              exchange:
                driver: "binance_futures"
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
            runtime = load_runtime_config(config_path=path, env={})
            self.assertEqual(runtime.backtest.data_source, "external")
            self.assertEqual(runtime.backtest.external.source_kind, "csv")
            self.assertEqual(runtime.backtest.external.root_path, "var/data/external/backtest")
            self.assertEqual(runtime.backtest.external.symbol_map["BTC/USDT"], "BTCUSDT.csv")
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
                driver: "binance_futures"
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
