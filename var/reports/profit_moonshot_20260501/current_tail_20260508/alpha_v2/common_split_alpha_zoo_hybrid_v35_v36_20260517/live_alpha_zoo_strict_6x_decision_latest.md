# Alpha Zoo strict 6x live decision artifact

- Candidate: `crypto_fx_alpha_zoo_state_calibrated` / `CryptoFxAlphaZooStateStrategy`
- Selected grid: `alpha_zoo_conservative_exit`
- Live decision: `selected_live_mode` / `strategy_class`
- Runtime overrides: leverage `6`, isolated margin, `target_allocation=0.10`, `window_seconds=3600`, `decision_cadence_seconds=3600`.
- Locked-OOS role: gate/report-only after train+validation candidate freeze; no locked-OOS selection violation in source artifact.
- Locked-OOS replay: return `0.205127`, MDD `0.067884`, Sharpe `1.772136`, Sortino `2.578776`, smart Sortino `2.414847`, Calmar `3.021741`, trades `365`, liquidations `0`, min buffer `9643.447509`.
- Live-equivalent test contract: live CLI param propagation plus MarketWindow-vs-MarketBatch strategy parity are required and covered by tests.
- Operator command: `uv run lq live --transport poll --decision-file /home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/live_alpha_zoo_strict_6x_decision_latest.json`
- Real mode note: this artifact does not bypass normal preflight, credentials, risk controls, or operator review.
