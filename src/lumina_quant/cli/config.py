"""lq config — runtime config inspection and validation commands.

Sub-commands
------------
lq config show      Print resolved RuntimeConfig as JSON (replaces BacktestConfigView
                    introspection and any hand-rolled config-dump scripts).
lq config validate  Load profile YAML through the full normalisation pipeline and surface
                    any ValueError at the exit gate — data.kinds typos, bad tokens, etc.

Both commands honour ``LQ_CONFIG_PATH`` (default: ``config.yaml``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


def _resolve_config_path(path_arg: str | None) -> str:
    return str(path_arg or os.environ.get("LQ_CONFIG_PATH", "config.yaml"))


def _rt_to_dict(rt) -> dict:
    """Convert RuntimeConfig to a JSON-serialisable dict."""
    try:
        return asdict(rt)
    except Exception:
        # Fallback: build dict manually for any non-standard dataclass shape.
        result: dict = {}
        for attr in vars(rt):
            value = getattr(rt, attr, None)
            try:
                json.dumps(value)
                result[attr] = value
            except (TypeError, ValueError):
                result[attr] = str(value)
        return result


def cmd_show(args: argparse.Namespace) -> int:
    """Print the fully resolved RuntimeConfig as JSON."""
    from lumina_quant.configuration import load_runtime_config

    path = _resolve_config_path(args.config)
    try:
        rt = load_runtime_config(path)
    except FileNotFoundError:
        from lumina_quant.configuration.schema import RuntimeConfig

        rt = RuntimeConfig()
        print(
            f"[warn] config file not found at '{path}'; showing schema defaults",
            file=sys.stderr,
        )
    payload = _rt_to_dict(rt)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Load the profile YAML and surface any validation errors at exit."""
    from lumina_quant.configuration import load_runtime_config, validate_runtime_config

    path = _resolve_config_path(args.config)
    try:
        rt = load_runtime_config(path)
    except FileNotFoundError:
        print(f"[error] config file not found: {path}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[error] config validation failed: {exc}", file=sys.stderr)
        return 1

    try:
        validate_runtime_config(rt, for_live=bool(args.live))
    except Exception as exc:
        print(f"[error] runtime validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"[ok] config valid: {Path(path).resolve()}")
    if bool(args.live):
        print("[ok] live-mode invariants passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lq config",
        description="Runtime config inspection and validation.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to config YAML (default: LQ_CONFIG_PATH env var or config.yaml).",
    )
    sub = parser.add_subparsers(dest="subcommand")

    _show = sub.add_parser("show", help="Print resolved RuntimeConfig as JSON.")
    _show.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Config YAML path.",
    )

    _val = sub.add_parser("validate", help="Validate config YAML; exit non-zero on any error.")
    _val.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Config YAML path.",
    )
    _val.add_argument(
        "--live",
        action="store_true",
        help="Also check live-mode invariants (validate_runtime_config(for_live=True)).",
    )

    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0
    if args.subcommand == "show":
        return cmd_show(args)
    if args.subcommand == "validate":
        return cmd_validate(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
