from __future__ import annotations

from lumina_quant.cli import dashboard


def test_build_dashboard_command_points_to_retired_stub() -> None:
    command = dashboard.build_dashboard_command()

    assert command[0]
    assert command[1].endswith('src/lumina_quant/dashboard/retired_stub.py')


def test_retired_stub_mentions_public_main() -> None:
    source = (dashboard.REPO_ROOT / 'src' / 'lumina_quant' / 'dashboard' / 'retired_stub.py').read_text(
        encoding='utf-8'
    )

    assert 'public/main' in source
    assert 'uv run lq backtest' in source
