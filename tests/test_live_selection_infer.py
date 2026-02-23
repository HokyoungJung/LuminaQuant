from lumina_quant.live_selection import infer_strategy_class_name


def test_infer_strategy_class_name_extended_catalog():
    assert infer_strategy_class_name("topcap_tsmom") == "TopCapTimeSeriesMomentumStrategy"
    assert infer_strategy_class_name("pair_xau_xag") == "PairTradingZScoreStrategy"
    assert infer_strategy_class_name("rolling_breakout_topcap") == "RollingBreakoutStrategy"
    assert infer_strategy_class_name("mean_reversion_std_topcap") == "MeanReversionStdStrategy"
    assert infer_strategy_class_name("vwap_reversion_topcap") == "VwapReversionStrategy"
    assert infer_strategy_class_name("lag_convergence_xau_xag") == "LagConvergenceStrategy"
    assert infer_strategy_class_name("bitcoin_buy_hold") == "BitcoinBuyHoldStrategy"
