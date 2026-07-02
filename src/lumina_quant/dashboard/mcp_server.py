"""Read-only MCP (Model Context Protocol) stdio bridge for LuminaQuant.

This module exposes a **strictly read-only** MCP surface over already-computed
research artifacts: backtest/overview summaries, factor IC-heatmap and
candidate-queue insights, alpha evidence, and the dashboard route contract.

Safety posture (enforced by structure, not convention)
------------------------------------------------------
* No trading verb is ever registered.  Every tool name is checked against
  :data:`FORBIDDEN_TOOL_SUBSTRINGS` at build time via :func:`assert_read_only`;
  a name containing ``order`` / ``place`` / ``submit`` / ``cancel`` / ``trade``
  / ``execute`` / ``buy`` / ``sell`` (etc.) raises before the server starts.
* The tool handlers import only read-only dashboard services; broker and
  order-gateway modules are never imported.
* The ``mcp`` runtime dependency is isolated behind the ``mcp`` extra and
  imported lazily inside :func:`build_mcp_server` only.  Introspection
  (:func:`list_tool_names`, ``--list-tools``) works with no ``mcp`` installed.
* The server is **opt-in**: nothing binds a transport at import time.  ``lq-mcp``
  (or ``lq mcp``) defaults to ``--list-tools`` and only serves stdio when
  ``--serve`` is passed explicitly.

The tool specs are plain data (:class:`ReadOnlyToolSpec`), so the read-only
contract can be asserted in tests without the MCP SDK present.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Any tool whose name contains one of these substrings is a mutating/trading
# verb and must never be exposed by this read-only bridge.
FORBIDDEN_TOOL_SUBSTRINGS: tuple[str, ...] = (
    "order",
    "place",
    "submit",
    "cancel",
    "trade",
    "execute",
    "buy",
    "sell",
    "liquidate",
    "close_position",
    "modify_position",
    "amend",
    "withdraw",
    "transfer",
    "route_order",
)


class MCPReadOnlyViolationError(RuntimeError):
    """Raised when a non-read-only tool name would be exposed."""


@dataclass(frozen=True, slots=True)
class ReadOnlyToolSpec:
    """Declarative spec for one read-only MCP tool (no MCP import required)."""

    name: str
    description: str
    handler: Callable[..., dict[str, Any]]


# ---------------------------------------------------------------------------
# Read-only tool handlers — each returns a JSON-serializable dashboard payload.
# All imports are deferred into the handler so that listing tools never pulls
# in Postgres / pandas machinery.
# ---------------------------------------------------------------------------


def _tool_backtest_overview() -> dict[str, Any]:
    """Read-only backtest/overview summary (safe empty when no DSN)."""
    from lumina_quant.dashboard.bridge import load_overview_payload

    return load_overview_payload(launch_mode="next")


def _tool_factor_insights(
    factor_ic_path: str = "",
    candidate_queue_path: str = "",
) -> dict[str, Any]:
    """Read-only factor IC-heatmap and candidate-queue insights."""
    from lumina_quant.dashboard.factor_insights_service import (
        load_factor_insights_payload,
    )

    return load_factor_insights_payload(
        factor_ic_path=factor_ic_path or None,
        candidate_queue_path=candidate_queue_path or None,
    )


def _tool_alpha_evidence(
    evidence_path: str = "",
    run_card_path: str = "",
    live_readiness_path: str = "",
) -> dict[str, Any]:
    """Read-only alpha-evidence / live-readiness summary."""
    from lumina_quant.dashboard.alpha_evidence_service import (
        load_alpha_evidence_payload,
    )

    return load_alpha_evidence_payload(
        evidence_path=evidence_path or None,
        run_card_path=run_card_path or None,
        live_readiness_path=live_readiness_path or None,
    )


def _tool_dashboard_routes() -> dict[str, Any]:
    """Read-only listing of the dashboard v2 route contract (diagnostic)."""
    from lumina_quant.dashboard.bridge import build_dashboard_bridge_contract_v2

    return build_dashboard_bridge_contract_v2().to_dict()


READ_ONLY_TOOL_SPECS: tuple[ReadOnlyToolSpec, ...] = (
    ReadOnlyToolSpec(
        name="get_backtest_overview",
        description=(
            "Return the read-only backtest/overview dashboard summary "
            "(recent runs, equity/drawdown curves). Safe-empty without a DSN."
        ),
        handler=_tool_backtest_overview,
    ),
    ReadOnlyToolSpec(
        name="get_factor_insights",
        description=(
            "Return the read-only factor IC-heatmap and candidate-queue "
            "insights payload from optional research artifacts."
        ),
        handler=_tool_factor_insights,
    ),
    ReadOnlyToolSpec(
        name="get_alpha_evidence",
        description=(
            "Return the read-only alpha-evidence and live-readiness summary. "
            "Advisory only; never enables real-money execution."
        ),
        handler=_tool_alpha_evidence,
    ),
    ReadOnlyToolSpec(
        name="list_dashboard_routes",
        description="Return the read-only dashboard v2 route contract (diagnostic).",
        handler=_tool_dashboard_routes,
    ),
)


def _is_forbidden(name: str) -> bool:
    lowered = str(name).lower()
    return any(token in lowered for token in FORBIDDEN_TOOL_SUBSTRINGS)


def assert_read_only(names: object) -> None:
    """Raise :class:`MCPReadOnlyViolationError` if any name is a mutating verb."""
    if isinstance(names, str):
        candidates = [names]
    else:
        candidates = [str(n) for n in names]  # type: ignore[union-attr]
    offenders = sorted({n for n in candidates if _is_forbidden(n)})
    if offenders:
        raise MCPReadOnlyViolationError(
            "read-only MCP bridge rejects mutating tool names: " + ", ".join(offenders)
        )


def list_tool_names() -> tuple[str, ...]:
    """Return the registered read-only tool names (no MCP import needed)."""
    return tuple(spec.name for spec in READ_ONLY_TOOL_SPECS)


def describe_tools() -> list[dict[str, str]]:
    """Return name+description for each read-only tool (no MCP import needed)."""
    return [{"name": spec.name, "description": spec.description} for spec in READ_ONLY_TOOL_SPECS]


def _import_fastmcp() -> Any:
    """Lazily import a FastMCP class from either ``mcp`` or ``fastmcp``."""
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]

        return FastMCP
    except Exception:  # pragma: no cover - exercised only when mcp installed differently
        try:
            from fastmcp import FastMCP  # type: ignore[import-not-found]

            return FastMCP
        except Exception as exc:  # pragma: no cover - no mcp runtime installed
            raise ModuleNotFoundError(
                "The MCP bridge requires the 'mcp' extra: install with "
                "`uv sync --extra mcp` (or `pip install 'lumina-quant[mcp]'`)."
            ) from exc


def build_mcp_server(name: str = "lumina-quant-readonly") -> Any:
    """Construct the FastMCP server with only read-only tools registered.

    Lazily imports the MCP SDK.  Read-only enforcement runs *before* any tool is
    registered, so a mutating verb can never reach the wire.
    """
    assert_read_only(list_tool_names())
    fastmcp_cls = _import_fastmcp()
    server = fastmcp_cls(name)
    for spec in READ_ONLY_TOOL_SPECS:
        # Guard again per-tool to keep the invariant local to registration.
        assert_read_only(spec.name)
        server.tool(name=spec.name, description=spec.description)(spec.handler)
    return server


def serve_stdio(name: str = "lumina-quant-readonly") -> int:
    """Build the read-only server and run it over stdio (blocking)."""
    server = build_mcp_server(name)
    run = getattr(server, "run", None)
    if run is None:  # pragma: no cover - defensive
        raise RuntimeError("MCP server object has no run() method")
    try:
        run(transport="stdio")
    except TypeError:  # pragma: no cover - older FastMCP signatures
        run()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``lq-mcp`` / ``lq mcp``.

    Defaults to ``--list-tools`` (introspection, no MCP runtime needed).  Pass
    ``--serve`` to actually start the stdio server (opt-in).
    """
    parser = argparse.ArgumentParser(
        prog="lq-mcp",
        description="Read-only LuminaQuant MCP stdio bridge (opt-in server).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the read-only MCP server over stdio (requires the 'mcp' extra).",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        dest="list_tools",
        help="Print the read-only tool catalog as JSON and exit (default).",
    )
    parser.add_argument(
        "--name",
        default="lumina-quant-readonly",
        help="Server name advertised to MCP clients.",
    )
    args = parser.parse_args(argv)

    if args.serve:
        # Re-affirm the read-only contract before binding a transport.
        assert_read_only(list_tool_names())
        return serve_stdio(args.name)

    # Default / --list-tools: pure introspection, no MCP import.
    assert_read_only(list_tool_names())
    print(
        json.dumps(
            {
                "artifact_kind": "mcp_readonly_tool_catalog",
                "server_name": args.name,
                "read_only": True,
                "real_money_execution_enabled": False,
                "tools": describe_tools(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "FORBIDDEN_TOOL_SUBSTRINGS",
    "READ_ONLY_TOOL_SPECS",
    "MCPReadOnlyViolationError",
    "ReadOnlyToolSpec",
    "assert_read_only",
    "build_mcp_server",
    "describe_tools",
    "list_tool_names",
    "serve_stdio",
]


if __name__ == "__main__":
    raise SystemExit(main())
