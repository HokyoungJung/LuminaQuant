"""Backfill derived funding-fee columns in futures feature-point parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def _ensure_columns(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    if "funding_fee_rate" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("funding_fee_rate"))
    if "funding_fee_quote_per_unit" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("funding_fee_quote_per_unit"))
    return out


def _rewrite_file(path: Path) -> dict[str, int | str]:
    frame = _ensure_columns(pl.read_parquet(path))
    original = frame

    updated = frame.with_columns(
        [
            pl.when(pl.col("funding_fee_rate").is_null())
            .then(pl.col("funding_rate"))
            .otherwise(pl.col("funding_fee_rate"))
            .cast(pl.Float64)
            .alias("funding_fee_rate"),
            pl.when(pl.col("funding_fee_quote_per_unit").is_null())
            .then(pl.col("funding_rate") * pl.col("funding_mark_price"))
            .otherwise(pl.col("funding_fee_quote_per_unit"))
            .cast(pl.Float64)
            .alias("funding_fee_quote_per_unit"),
        ]
    )

    changed = original.columns != updated.columns or not original.equals(updated, null_equal=True)
    if not changed:
        return {"file": str(path), "updated": 0, "rows": int(updated.height)}

    tmp_path = path.with_suffix(".tmp.parquet")
    updated.write_parquet(tmp_path, compression="zstd", statistics=True)
    tmp_path.replace(path)
    return {"file": str(path), "updated": 1, "rows": int(updated.height)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill funding-fee derived columns in feature-point parquet files."
    )
    parser.add_argument("--db-path", default="data/market_parquet")
    args = parser.parse_args()

    root = Path(str(args.db_path)).expanduser() / "feature_points"
    files = sorted(root.glob("exchange=*/symbol=*/date=*/*.parquet"))

    results = [_rewrite_file(path) for path in files]
    updated_files = sum(int(item["updated"]) for item in results)
    payload = {
        "db_path": str(args.db_path),
        "feature_files": len(files),
        "updated_files": updated_files,
        "files": results,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
