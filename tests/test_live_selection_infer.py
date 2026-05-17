from lumina_quant.live_selection import extract_live_decision_config, infer_strategy_class_name


def test_infer_strategy_class_name_extended_catalog():
    assert infer_strategy_class_name("topcap_tsmom") == "TopCapTimeSeriesMomentumStrategy"
    assert infer_strategy_class_name("pair_xau_xag") == "PairTradingZScoreStrategy"
    assert infer_strategy_class_name("rolling_breakout_topcap") == "RollingBreakoutStrategy"
    assert infer_strategy_class_name("mean_reversion_std_topcap") == "MeanReversionStdStrategy"
    assert infer_strategy_class_name("vwap_reversion_topcap") == "VwapReversionStrategy"
    assert infer_strategy_class_name("lag_convergence_xau_xag") == "LagConvergenceStrategy"
    assert infer_strategy_class_name("bitcoin_buy_hold") == "BitcoinBuyHoldStrategy"
    assert infer_strategy_class_name("crypto_fx_alpha_zoo_state_calibrated") == "CryptoFxAlphaZooStateStrategy"
    assert infer_strategy_class_name("alpha_zoo_conservative_exit") == "CryptoFxAlphaZooStateStrategy"
    assert (
        infer_strategy_class_name("alpha_zoo_strict_6x_common_split_reselected:alpha_zoo_conservative_exit")
        == "CryptoFxAlphaZooStateStrategy"
    )
    assert infer_strategy_class_name("panic_rebound_mr_5m") == "PanicReboundMeanReversionStrategy"
    assert (
        infer_strategy_class_name("session_filtered_pair_carry_1h")
        == "SessionFilteredPairCarryStrategy"
    )
    assert infer_strategy_class_name("profit_moonshot_trend_1h_balanced") == "ProfitMoonshotTrendStrategy"
    assert (
        infer_strategy_class_name("profit_moonshot_breakout_1h_expansion")
        == "ProfitMoonshotBreakoutStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_reversion_1h_shock")
        == "ProfitMoonshotReversionStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_perp_crowding_carry")
        == "PerpCrowdingCarryStrategy"
    )
    assert (
        infer_strategy_class_name("dfse_15m_top5_exhaustion_plus_flow")
        == "DerivativesFlowSqueezeStrategy"
    )
    assert (
        infer_strategy_class_name("derivatives_flow_squeeze_mode")
        == "DerivativesFlowSqueezeStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_derivatives_taker_flow_mode")
        == "DerivativesFlowSqueezeStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_derivatives_taker_flow_sparse_mode")
        == "DerivativesFlowSqueezeStrategy"
    )


def test_infer_strategy_class_name_leadlag_slow_diffusion_mode():
    assert (
        infer_strategy_class_name("profit_moonshot_leadlag_slow_diffusion_mode")
        == "CrossCryptoSlowDiffusionStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_leadlag_slow_diffusion_ensemble_mode")
        == "CrossCryptoSlowDiffusionStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_leadlag_slow_diffusion_sol_eth_mode")
        == "CrossCryptoSlowDiffusionStrategy"
    )


def test_infer_strategy_class_name_hourly_shock_reversion_mode():
    assert (
        infer_strategy_class_name("profit_moonshot_hourly_shock_reversion_eth_mode")
        == "HourlyShockReversionStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_hourly_shock_reversion_eth_12h_mode")
        == "HourlyShockReversionStrategy"
    )


def test_infer_strategy_class_name_precious_metal_pair_mode():
    assert (
        infer_strategy_class_name("profit_moonshot_precious_metal_pair_aggressive_mode")
        == "TimeframePairZScoreReversionStrategy"
    )


def test_infer_taker_flow_exhaustion_strategy_name() -> None:
    assert (
        infer_strategy_class_name("profit_moonshot_taker_flow_exhaustion_eth_mode")
        == "TakerFlowExhaustionReversalStrategy"
    )


def test_extract_live_decision_config_preserves_alpha_zoo_runtime_overrides() -> None:
    payload = {
        "decision": "selected_live_mode",
        "selected_mode": "crypto_fx_alpha_zoo_state_calibrated",
        "strategy_name": "CryptoFxAlphaZooStateStrategy",
        "symbols": ["btc/usdt", "eth/usdt"],
        "strategy_timeframe": "1h",
        "strategy_params": {
            "entry_threshold": 0.95,
            "calibrated_edges": {"default:LONG": 1.0},
            "decision_cadence_seconds": 3600,
        },
        "leverage": 6,
        "target_allocation": 0.10,
        "window_seconds": 3600,
        "ingest_window_seconds": 3600,
        "decision_cadence_seconds": 3600,
    }

    config = extract_live_decision_config(payload)

    assert config["target_kind"] == "strategy_class"
    assert config["strategy_name"] == "CryptoFxAlphaZooStateStrategy"
    assert config["symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert config["strategy_timeframe"] == "1h"
    assert config["strategy_params"]["calibrated_edges"] == {"default:LONG": 1.0}
    assert config["leverage"] == 6
    assert config["exchange"]["leverage"] == 6
    assert config["target_allocation"] == 0.10
    assert config["window_seconds"] == 3600
    assert config["ingest_window_seconds"] == 3600
    assert config["decision_cadence_seconds"] == 3600


def test_extract_live_decision_config_preserves_alpha_zoo_7x_isolated_overrides() -> None:
    payload = {
        "decision": "selected_live_mode",
        "selected_mode": "alpha_zoo_fast_residual",
        "strategy_name": "CryptoFxAlphaZooStateStrategy",
        "symbols": ["btc/usdt", "eth/usdt", "sol/usdt"],
        "strategy_timeframe": "1h",
        "strategy_params": {
            "entry_threshold": 0.9,
            "exit_threshold": 0.25,
            "fast_lookback_bars": 2,
            "slow_lookback_bars": 18,
            "history_window": 72,
        },
        "exchange": {
            "driver": "binance_futures",
            "name": "binance",
            "market_type": "future",
            "position_mode": "HEDGE",
            "margin_mode": "isolated",
            "leverage": 7,
        },
        "target_allocation": 0.15,
        "window_seconds": 3600,
        "ingest_window_seconds": 3600,
        "decision_cadence_seconds": 3600,
    }

    config = extract_live_decision_config(payload)

    assert config["target_kind"] == "strategy_class"
    assert config["strategy_name"] == "CryptoFxAlphaZooStateStrategy"
    assert config["symbols"] == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert config["exchange"]["margin_mode"] == "isolated"
    assert config["leverage"] == 7
    assert config["exchange"]["leverage"] == 7
    assert config["target_allocation"] == 0.15
