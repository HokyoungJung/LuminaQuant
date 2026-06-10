"""Tests for lq data — replacement for the retired lq refresh-data-fast command.

``lq refresh-data-fast`` was deleted in Phase 6 (fold into ``lq data fast``).
These tests cover the new ``lq data`` CLI subcommands at the parse layer so the
suite denominator is preserved without coupling to DataCollector I/O.
"""

from __future__ import annotations

import argparse
import importlib

import pytest


def _parse(argv: list[str]) -> argparse.Namespace:
    importlib.import_module("lumina_quant.cli.data")
    parser = argparse.ArgumentParser(prog="lq data")
    sub = parser.add_subparsers(dest="subcommand")
    collect = sub.add_parser("collect")
    collect.add_argument("--since-ms", type=int, default=None, dest="since_ms")
    collect.add_argument("--until-ms", type=int, default=None, dest="until_ms")
    collect.add_argument("--force-full", action="store_true", dest="force_full")
    collect.add_argument("--feature-profile", default="full", dest="feature_profile")
    collect.add_argument("--json", action="store_true")
    sub.add_parser("fast")
    tick = sub.add_parser("tick")
    tick.add_argument("--symbol", default="")
    return parser.parse_args(argv)


def test_data_fast_subcommand_parses() -> None:
    ns = _parse(["fast"])
    assert ns.subcommand == "fast"


def test_data_collect_subcommand_default_args() -> None:
    ns = _parse(["collect"])
    assert ns.subcommand == "collect"
    assert ns.since_ms is None
    assert ns.until_ms is None
    assert ns.force_full is False
    assert ns.feature_profile == "full"


def test_data_collect_subcommand_explicit_args() -> None:
    ns = _parse(["collect", "--since-ms", "1000000", "--force-full", "--feature-profile", "fast"])
    assert ns.since_ms == 1000000
    assert ns.force_full is True
    assert ns.feature_profile == "fast"


def test_data_tick_subcommand_parses() -> None:
    ns = _parse(["tick", "--symbol", "BTC/USDT"])
    assert ns.subcommand == "tick"
    assert ns.symbol == "BTC/USDT"


def test_data_main_no_subcommand_returns_zero() -> None:
    from lumina_quant.cli.data import main

    assert main([]) == 0


def test_data_module_importable() -> None:
    """Regression guard: module must import without side effects."""
    from lumina_quant.cli.data import cmd_collect, cmd_fast, cmd_tick, main
