from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Touch a stop-file for graceful live shutdown.")
    parser.add_argument("--stop-file", default="/tmp/lq.stop")
    args = parser.parse_args(argv)

    stop_path = Path(args.stop_file).expanduser()
    existed_before = stop_path.exists()
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.touch()

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "live_stop_request",
        "stop_file": str(stop_path),
        "existed_before": existed_before,
        "exists_after": stop_path.exists(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
