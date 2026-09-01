# Context Snapshot — web-grounded autonomous research expansion

- task statement: Continue the autonomous LuminaQuant portfolio research loop as a web-grounded autonomous researcher. Audit current strategy, indicator, alpha, tuning, portfolio, optimization, and backtest implementations; research stronger methods from external primary sources; run bounded experiments under the 8 GiB total-session cap; keep winners and discard losers.
- desired outcome:
  1. preserve the current best incumbent while continuing to search for materially better challengers;
  2. generate a durable pipeline for web-grounded idea intake -> tuning -> backtest -> keep/discard;
  3. use team mode for execution/backtest tuning lanes and Ralph for final verification / bug-fixing / cleanliness;
  4. avoid idle pauses and continue automatically until interrupted.
- known facts/evidence:
  - current best incumbent on 2026-03-15 is a 3-sleeve portfolio:
    - pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 @ 60.0%
    - composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 @ 25.4390%
    - topcap_tsmom_1h_balanced_16_4_0.015 @ 14.5610%
  - current incumbent locked-OOS metrics:
    - total_return 5.5960%
    - sharpe 3.447
    - max_drawdown 1.4277%
    - promotion_score 13.4725
  - dynamic and overlay challengers were rerun against the rolled 3-sleeve incumbent and both failed to clear promotion.
  - targeted 4th-sleeve probes (rolling/regime/carry/alt-pair) also failed to beat the current 3-sleeve incumbent.
  - existing PRD/test-spec planning gate artifacts already exist:
    - .omx/plans/prd-autonomous-portfolio-research-loop.md
    - .omx/plans/test-spec-autonomous-portfolio-research-loop.md
  - tmux+omx team runtime is available in this session.
- constraints:
  - keep total memory across the whole session under 8 GiB.
  - one heavy lane at a time.
  - reuse exact-window registry / existing followup artifacts; do not add a second scheduler.
  - continue automatically without asking the user whether to keep going.
  - use primary sources when browsing the web for external research.
- unknowns/open questions:
  - whether web-grounded ideas such as residual momentum, crash-aware momentum gating, carry hybrids, or improved pair state models can beat the current 3-sleeve incumbent in this codebase.
  - whether a new family requires lightweight implementation before it can be evaluated.
  - whether current portfolio comparison/decision plumbing needs additional cleanup before private push.
- likely codebase touchpoints:
  - src/lumina_quant/strategy_factory/
  - src/lumina_quant/eval/
  - scripts/run_portfolio_optimization.py
  - scripts/research/run_causal_dynamic_portfolio.py
  - scripts/research/run_causal_overlay_portfolio.py
  - var/reports/exact_window_backtests/followup_status/
