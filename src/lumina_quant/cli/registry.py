"""lq registry — plugin registry inspection commands.

Sub-commands
------------
lq registry list    Enumerate PluginRegistry entries (strategies, adapters, etc.)
                    registered for the current config profile.
"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_list(args: argparse.Namespace) -> int:
    """Print all registered plugin registry entries as JSON."""
    try:
        from lumina_quant.strategies import registry
    except Exception as exc:
        print(f"[error] failed to load strategy registry: {exc}", file=sys.stderr)
        return 1

    entries: dict = {}

    # Strategy map
    try:
        strategy_map = registry.get_strategy_map()
        entries["strategies"] = sorted(strategy_map.keys())
    except Exception:
        entries["strategies"] = []

    # Live strategy map (strategies that opt in to live execution)
    try:
        live_map = (
            registry.get_live_strategy_map(include_opt_in=True)
            if hasattr(registry, "get_live_strategy_map")
            else registry.get_strategy_map()
        )
        entries["live_strategies"] = sorted(live_map.keys())
    except Exception:
        entries["live_strategies"] = []

    # Decorator-registered indicator / portfolio plugins (GLOBAL_REGISTRY).
    # Triggering get_strategy_map above already ran strategy package
    # auto-discovery, so any dropped @register modules are now reflected here.
    try:
        from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY

        entries["indicators"] = sorted(
            n for n in GLOBAL_REGISTRY.list_names("indicator") if not n.startswith("_")
        )
        entries["portfolios"] = sorted(
            n for n in GLOBAL_REGISTRY.list_names("portfolio") if not n.startswith("_")
        )
    except Exception:
        entries["indicators"] = []
        entries["portfolios"] = []

    # Default strategy name
    try:
        entries["default_strategy"] = getattr(registry, "DEFAULT_STRATEGY_NAME", None)
    except Exception:
        entries["default_strategy"] = None

    # Registry metadata if available
    try:
        entries["registry_version"] = getattr(registry, "REGISTRY_VERSION", None)
        entries["registry_source"] = getattr(registry, "__module__", None)
    except Exception:
        pass

    if bool(getattr(args, "json", False)):
        print(json.dumps(entries, indent=2, sort_keys=True))
    else:
        print(f"Strategies ({len(entries['strategies'])}):")
        for name in entries["strategies"]:
            marker = " [live]" if name in entries.get("live_strategies", []) else ""
            print(f"  {name}{marker}")
        if entries.get("indicators"):
            print(f"Indicators ({len(entries['indicators'])}):")
            for name in entries["indicators"]:
                print(f"  {name}")
        if entries.get("portfolios"):
            print(f"Portfolios ({len(entries['portfolios'])}):")
            for name in entries["portfolios"]:
                print(f"  {name}")
        if entries.get("default_strategy"):
            print(f"Default: {entries['default_strategy']}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lq registry",
        description="Plugin registry inspection.",
    )
    sub = parser.add_subparsers(dest="subcommand")

    _list = sub.add_parser("list", help="Enumerate registered strategies and adapters.")
    _list.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON.",
    )

    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0
    if args.subcommand == "list":
        return cmd_list(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
