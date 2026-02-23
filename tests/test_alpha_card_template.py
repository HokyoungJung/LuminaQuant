from __future__ import annotations

import os
import sys
import tempfile
import textwrap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from generate_alpha_card_template import generate_alpha_card


def test_generate_alpha_card_uses_strategy_profile_and_writes_file():
    yaml_text = textwrap.dedent(
        """
        trading:
          symbols: ["BTC/USDT", "ETH/USDT"]
          timeframe: "5m"
        optimization:
          strategy: "RsiStrategy"
        live:
          mode: "paper"
          poll_interval: 2
          order_timeout: 10
          reconciliation_interval_sec: 30
          exchange:
            driver: "ccxt"
            name: "binance"
            market_type: "future"
            position_mode: "HEDGE"
            margin_mode: "isolated"
            leverage: 2
        promotion_gate:
          days: 14
          strategy_profiles:
            RsiStrategy:
              days: 21
              max_order_rejects: 1
              max_order_timeouts: 1
              max_reconciliation_alerts: 0
              max_critical_errors: 0
              require_alpha_card: true
        """
    ).strip()

    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.yaml")
        output_path = os.path.join(tmp, "alpha_card.md")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(yaml_text)

        written_path, markdown = generate_alpha_card(
            config_path=config_path,
            strategy_name="RsiStrategy",
            output_path=output_path,
            overwrite=False,
        )

        assert written_path == output_path
        assert os.path.exists(output_path)
        assert "# Alpha Card - RsiStrategy" in markdown
        assert "- Window days: 21" in markdown
        assert "- Require alpha card for gate: True" in markdown


def test_generate_alpha_card_requires_overwrite_flag_when_file_exists():
    yaml_text = textwrap.dedent(
        """
        trading:
          symbols: ["BTC/USDT"]
        optimization:
          strategy: "RsiStrategy"
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

    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.yaml")
        output_path = os.path.join(tmp, "alpha_card.md")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(yaml_text)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("existing")

        raised = False
        try:
            generate_alpha_card(
                config_path=config_path,
                strategy_name="RsiStrategy",
                output_path=output_path,
                overwrite=False,
            )
        except FileExistsError:
            raised = True

        assert raised is True
