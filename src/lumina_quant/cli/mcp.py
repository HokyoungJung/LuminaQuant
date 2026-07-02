"""``lq mcp`` — read-only MCP stdio bridge (opt-in).

Thin delegate to :func:`lumina_quant.dashboard.mcp_server.main`.  Defaults to
listing the read-only tool catalog; ``--serve`` starts the stdio server (which
requires the ``mcp`` extra).  No trading verbs are ever exposed.
"""

from __future__ import annotations

from lumina_quant.dashboard.mcp_server import main as _mcp_main


def main(argv: list[str] | None = None) -> int:
    return _mcp_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
