from __future__ import annotations

from types import ModuleType

from lumina_quant.cli import main as cli_main


def test_main_does_not_import_command_handlers_without_selection(monkeypatch):
    def fail_import(_name: str):
        raise AssertionError(f"Unexpected import during root CLI parse: {_name}")

    monkeypatch.setattr(cli_main, "import_module", fail_import)
    assert cli_main.main([]) == 0


def test_main_loads_only_selected_command_handler(monkeypatch):
    calls: list[str] = []

    def fake_import(name: str):
        calls.append(name)
        module = ModuleType(name)

        def _main(args=None):
            return 0

        module.main = _main
        return module

    monkeypatch.setattr(cli_main, "import_module", fake_import)
    assert cli_main.main(["backtest", "--dry-run"]) == 0
    assert calls == ["lumina_quant.cli.backtest"]


def test_main_exposes_lazy_module_attributes_for_compatibility(monkeypatch):
    calls: list[str] = []

    def fake_import(name: str):
        calls.append(name)
        module = ModuleType(name)
        module.main = lambda args=None: 0
        return module

    monkeypatch.delitem(cli_main.__dict__, "exact_window", raising=False)
    monkeypatch.setattr(cli_main, "import_module", fake_import)
    module = cli_main.exact_window
    assert module.main([]) == 0
    assert calls == ["lumina_quant.cli.exact_window"]
