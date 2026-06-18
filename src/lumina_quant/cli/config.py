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


#: Field names whose values are secrets and must be masked before display.
#: Matching is case-insensitive on the exact key name.
_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "secret_key",
        "telegram_bot_token",
    }
)

#: Field names whose values are connection strings that may embed a password
#: (``scheme://user:password@host/db``); the password component is scrubbed.
_DSN_FIELD_NAMES = frozenset(
    {
        "postgres_dsn",
        "dsn",
    }
)


def _mask_secret(value) -> str:
    """Mask a secret value, preserving the last 4 characters for identification."""
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 4:
        return "***"
    return "***" + text[-4:]


def _scrub_dsn_password(value) -> str:
    """Replace the password component of a ``scheme://user:password@host`` DSN."""
    text = str(value or "")
    if "@" not in text or "://" not in text:
        return text
    scheme, _, rest = text.partition("://")
    creds, sep, hostpart = rest.partition("@")
    if sep and ":" in creds:
        user, _, _ = creds.partition(":")
        creds = f"{user}:***"
    return f"{scheme}://{creds}{sep}{hostpart}"


def _redact_secrets(value):
    """Recursively mask secret-bearing fields in a JSON-serialisable structure."""
    if isinstance(value, dict):
        redacted: dict = {}
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in _SECRET_FIELD_NAMES:
                redacted[key] = None if item is None else _mask_secret(item)
            elif key_l in _DSN_FIELD_NAMES:
                redacted[key] = _scrub_dsn_password(item) if item else item
            else:
                redacted[key] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_secrets(item) for item in value]
    return value


def _rt_to_dict(rt) -> dict:
    """Convert RuntimeConfig to a JSON-serialisable dict with secrets redacted."""
    try:
        payload = asdict(rt)
    except Exception:
        # Fallback: build dict manually for any non-standard dataclass shape.
        payload = {}
        for attr in vars(rt):
            value = getattr(rt, attr, None)
            try:
                json.dumps(value)
                payload[attr] = value
            except TypeError, ValueError:
                payload[attr] = str(value)
    return _redact_secrets(payload)


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
