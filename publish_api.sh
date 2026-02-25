#!/bin/bash

set -euo pipefail

uv run python scripts/publish_public_pr.py "$@"
