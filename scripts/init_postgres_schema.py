"""Initialize LuminaQuant PostgreSQL runtime schema."""

from __future__ import annotations

import argparse
import os

from lumina_quant.postgres_state import SCHEMA_SQL, PostgresStateRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize LuminaQuant Postgres schema for local-only runtime state."
    )
    parser.add_argument(
        "--dsn",
        default="",
        help="Postgres DSN (fallback: LQ_POSTGRES_DSN env var).",
    )
    parser.add_argument(
        "--print-ddl",
        action="store_true",
        help="Print schema DDL and exit without applying it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_ddl:
        for statement in SCHEMA_SQL:
            print(statement.strip().rstrip(";"))
            print(";")
        return 0

    dsn = str(args.dsn or "").strip() or str(os.getenv("LQ_POSTGRES_DSN", "")).strip()
    if not dsn:
        raise SystemExit("Postgres DSN is required. Set --dsn or LQ_POSTGRES_DSN.")

    repo = PostgresStateRepository(dsn=dsn)
    repo.initialize_schema()
    print("Postgres schema initialized for runs/equity/orders/fills/positions/workflow tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
