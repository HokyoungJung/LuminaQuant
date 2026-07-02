# G003 Point-in-Time External Prior Registry

- Generated: `2026-07-02T14:21:38Z`
- Registry SHA256: `32fe1cb8e9ce809c9f94959c1a6348f212f4ef5cd48c0a49f40c7ef4aacb592b`
- Sources: 9; retrieved: 9; diagnostic-only: 7
- Live shadow / paper / real-money execution: **false / blocked**

## Policy

External sources are research design priors only. Candidate definitions must be frozen in G004 before evaluation. Missing point-in-time proof demotes historical metric use to `research_diagnostic_only`; imported external performance claims are not accepted as LuminaQuant evidence.

| source_id | status | published | prior_available_at | class | derived specs |
|---|---|---|---|---|---|
| `arxiv_deepm_regime_robust_macro` | retrieved (200) | `2026-01-09T00:00:00Z` | `2026-01-10T00:00:00Z` | `research_diagnostic_only` | strict_lagged_cross_sectional_attention_proxy, macro_graph_sector_prior, worst_window_robust_utility_penalty |
| `arxiv_trendfolios_multi_asset_trend` | retrieved (200) | `2025-06-11T00:00:00Z` | `2025-06-12T00:00:00Z` | `research_diagnostic_only` | multi_asset_trend_signal, inverse_volatility_weighting, drawdown_volatility_risk_budget |
| `arxiv_hmm_rl_regime_allocation` | retrieved (200) | `2026-05-27T00:00:00Z` | `2026-05-28T00:00:00Z` | `research_diagnostic_only` | three_state_hmm_regime_proxy, one_day_lagged_regime_allocation, stress_regime_defensive_weighting |
| `arxiv_adaptivetrend_crypto` | retrieved (200) | `2026-02-12T00:00:00Z` | `2026-02-13T00:00:00Z` | `research_diagnostic_only` | adaptive_trailing_stop_volatility_regime, rolling_sharpe_asset_selection, asymmetric_long_short_allocation |
| `binance_usdm_exchange_info_docs` | retrieved (200) | `2026-07-02T06:58:44Z` | `2026-07-02T06:58:44Z` | `schema_metadata_only_not_alpha_evidence` | exchange_status_filter, contract_type_filter, tradfi_underlying_subtype_filter |
| `binance_usdm_funding_rate_docs` | retrieved (200) | `2026-07-02T06:58:44Z` | `2026-07-02T06:58:44Z` | `schema_metadata_only_not_alpha_evidence` | funding_rate, funding_mark_price, funding_fee_quote_per_unit |
| `yahoo_chart_spy_research_probe` | retrieved (200) | `None` | `None` | `research_diagnostic_only` | tradfi_proxy_return_diagnostic |
| `stooq_spy_daily_research_probe` | retrieved (200) | `None` | `None` | `research_diagnostic_only` | tradfi_proxy_return_diagnostic |
| `sec_companyfacts_aapl_probe` | retrieved (200) | `None` | `None` | `research_diagnostic_only` | fundamental_release_lag_diagnostic |
