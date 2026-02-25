from __future__ import annotations

from pathlib import Path


def test_run_bot_uses_uv_runtime():
    script_path = Path(__file__).resolve().parents[1] / "run_bot.sh"
    content = script_path.read_text(encoding="utf-8")

    assert "uv run python run_live.py" in content
    assert "python3 run_live.py" not in content
