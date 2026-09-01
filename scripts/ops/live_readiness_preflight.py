from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumina_quant.live.readiness_policy import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DECISION_JSON,
    DEFAULT_PREFLIGHT_STALE_MINUTES,
    DEFAULT_REFRESH_JSON,
    build_live_readiness_payload,
)


def build_preflight_payload(
    *,
    config_path: Path,
    refresh_json: Path,
    decision_json: Path,
    stale_minutes: int,
) -> dict:
    return build_live_readiness_payload(
        config_path=config_path,
        refresh_json=refresh_json,
        decision_json=decision_json,
        stale_minutes=stale_minutes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check paper/live readiness preflight.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--refresh-json", default=str(DEFAULT_REFRESH_JSON))
    parser.add_argument("--decision-json", default=str(DEFAULT_DECISION_JSON))
    parser.add_argument("--stale-minutes", type=int, default=DEFAULT_PREFLIGHT_STALE_MINUTES)
    args = parser.parse_args(argv)

    payload = build_preflight_payload(
        config_path=Path(args.config),
        refresh_json=Path(args.refresh_json),
        decision_json=Path(args.decision_json),
        stale_minutes=max(1, int(args.stale_minutes)),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
