from types import SimpleNamespace

from lumina_quant.cli import backtest as backtest_cli
from lumina_quant.strategies import artifact_portfolio_mode as portfolio_modes
from lumina_quant.live_selection import (
    normalize_portfolio_mode_reference,
    resolve_portfolio_mode_runtime_config,
    supports_live_portfolio_mode,
)
from lumina_quant.strategies.artifact_portfolio_mode import (
    PortfolioModeDefinition,
    resolve_portfolio_mode_definition,
)


def test_live_selection_accepts_bare_and_wrapped_portfolio_modes() -> None:
    assert (
        normalize_portfolio_mode_reference("ArtifactPortfolioModeStrategy[static_blend_76_24]")
        == "static_blend_76_24"
    )
    assert supports_live_portfolio_mode("static_blend_76_24")
    assert supports_live_portfolio_mode("ArtifactPortfolioModeStrategy[state_vwap_pair]")


def test_source_sleeve_modes_expand_as_live_portfolio_modes(monkeypatch, tmp_path) -> None:
    blend_path = tmp_path / "blend.json"
    blend_path.write_text(
        """
        {
          "weights": [
            {
              "candidate_id": "pair_fixture",
              "name": "pair_fixture",
              "weight_share": 1.0,
              "strategy_class": "PairSpreadZScoreStrategy",
              "symbols": ["BNB/USDT", "TRX/USDT"],
              "params": {"z_entry": 2.0, "z_exit": 0.5}
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(portfolio_modes, "REFRESHED_BLEND_PATH", blend_path)

    definition = resolve_portfolio_mode_definition("static_blend_76_24")

    assert definition.components
    assert "BNB/USDT" in definition.symbols
    assert "TRX/USDT" in definition.symbols


def test_backtest_resolves_portfolio_mode_through_live_runtime(monkeypatch) -> None:
    class _ModeStrategy:
        __name__ = "ArtifactPortfolioModeStrategy"

    monkeypatch.setattr(backtest_cli, "ArtifactPortfolioModeStrategy", _ModeStrategy)
    monkeypatch.setattr(
        backtest_cli,
        "resolve_portfolio_mode_runtime_config",
        lambda mode: {
            "portfolio_mode": mode,
            "strategy_name": f"ArtifactPortfolioModeStrategy[{mode}]",
            "strategy_params": {"portfolio_mode": mode},
            "symbols": ["BNB/USDT", "TRX/USDT"],
            "cash_weight": 0.0,
        },
    )
    monkeypatch.setattr(
        backtest_cli,
        "_get_strategy_registry",
        lambda: SimpleNamespace(DEFAULT_STRATEGY_NAME="unused"),
    )

    setup = backtest_cli._resolve_strategy_setup_detail(
        log=False,
        portfolio_mode="state_vwap_pair",
    )

    assert setup.strategy_cls is _ModeStrategy
    assert setup.strategy_name == "ArtifactPortfolioModeStrategy[state_vwap_pair]"
    assert setup.strategy_params == {"portfolio_mode": "state_vwap_pair"}
    assert setup.symbol_list == ["BNB/USDT", "TRX/USDT"]


def test_runtime_config_returns_live_symbols_for_pair_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        portfolio_modes,
        "resolve_portfolio_mode_definition",
        lambda mode: PortfolioModeDefinition(
            portfolio_mode=mode,
            components=(),
            cash_weight=0.0,
            source_artifacts={},
            watch_symbols=("BNB/USDT", "TRX/USDT"),
        ),
    )

    config = resolve_portfolio_mode_runtime_config("state_vwap_pair")

    assert config["strategy_name"] == "ArtifactPortfolioModeStrategy[state_vwap_pair]"
    assert config["strategy_params"] == {"portfolio_mode": "state_vwap_pair"}
    assert config["symbols"] == ["BNB/USDT", "TRX/USDT"]
