from __future__ import annotations

from lumina_quant.cli import main as cli_main


def test_backtest_help_has_no_strategy_setup_noise(capsys):
    assert cli_main.main(["backtest", "--help"]) == 0
    captured = capsys.readouterr()
    assert "Optimized params not found" not in captured.out
    assert "Run LuminaQuant backtest." in captured.out


def test_optimize_help_has_no_optuna_import_warning(capsys):
    assert cli_main.main(["optimize", "--help"]) == 0
    captured = capsys.readouterr()
    assert "Optuna not found" not in captured.out
    assert "Run LuminaQuant walk-forward optimization." in captured.out


def test_backtest_help_does_not_force_strategy_registry_loading(monkeypatch):
    from lumina_quant.cli import backtest as backtest_cli

    monkeypatch.setattr(
        backtest_cli,
        "_get_strategy_registry",
        lambda: (_ for _ in ()).throw(
            AssertionError("strategy registry loaded during backtest help")
        ),
    )
    assert cli_main.main(["backtest", "--help"]) == 0


def test_optimize_help_does_not_force_strategy_registry_loading(monkeypatch):
    from lumina_quant.cli import optimize as optimize_cli

    monkeypatch.setattr(
        optimize_cli,
        "_get_strategy_registry",
        lambda: (_ for _ in ()).throw(
            AssertionError("strategy registry loaded during optimize help")
        ),
    )
    assert cli_main.main(["optimize", "--help"]) == 0


def test_live_help_runs_without_name_errors(capsys):
    assert cli_main.main(["live", "--help"]) == 0
    captured = capsys.readouterr()
    assert "Run LuminaQuant live trader." in captured.out


def test_live_help_does_not_force_runtime_or_strategy_loading(monkeypatch):
    from lumina_quant.cli import live as live_cli

    monkeypatch.setattr(
        live_cli,
        "_strategy_helpers",
        lambda: (_ for _ in ()).throw(AssertionError("strategy helpers loaded during help")),
    )
    monkeypatch.setattr(
        live_cli,
        "build_live_runtime_contract",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("runtime contract loaded during help")
        ),
    )
    assert cli_main.main(["live", "--help"]) == 0
