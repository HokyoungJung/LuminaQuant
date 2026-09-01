# Session handoff — reboot validation rerun completed

## Current checkpoint
- repo: `/home/hoky/Quants-agent/LuminaQuant`
- branch: `private-main`
- code commit before this rerun: `8fab962`
- current date context: `2026-04-16`

## What was re-run sequentially
All heavy work was re-run **sequentially only** with:
- `POLARS_MAX_THREADS=1`
- `RAYON_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `LQ_BACKTEST_LOW_MEMORY=1`
- `LQ_AUTO_COLLECT_DB=0`

Executed pipeline:
1. `run_current_switch_validation_reboot.py`
2. `write_portfolio_operating_switch.py --artifact-profile reboot_validation`
3. `tune_hybrid_online_portfolio.py`
4. `optuna_tune_hybrid_online_portfolio.py --objective-profile live_guarded --n-trials 24`
5. `optuna_tune_hybrid_online_portfolio.py --objective-profile train_aware_guarded --n-trials 24`
6. `run_hybrid_online_portfolio.py --config-json .../portfolio_hybrid_online_optuna_current/hybrid_online_optuna_latest.json`
7. `write_portfolio_operating_switch.py --artifact-profile reboot_validation`
8. `write_portfolio_operating_playbook.py`
9. `write_portfolio_master_scoreboard.py`

## Memory / OOM evidence
Peak RSS stayed well below the 8 GiB cap:
- reboot switch validation rebuild: `~404 MiB`
- reboot switch writer: `~205 MiB`
- curated hybrid tuning: `~70 MiB`
- optuna live_guarded (24 trials): `~89 MiB`
- optuna train_aware_guarded (24 trials): `~89 MiB`
- hybrid online portfolio refresh: `~72 MiB`

No user python/uv heavy process was left running after completion.

## Current validated result
### Reboot split contract
- train: `2025-01-01` ~ `2025-12-31`
- val: `2026-01-01` ~ `2026-02-28`
- oos: `2026-03-01` ~ latest
- warmup_ratio / warmup_days / online_start: `0.60` / `255` / `2025-09-13`

### Current reboot switch state
Artifact:
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/refreshed_operating_switch_current/portfolio_operating_switch_latest.md`

State:
- favored_group: `mixed`
- confidence: `0.0`
- trend: `bullish`
- breadth: `broad`
- volatility: `calm`
- pair_liquidity: `normal`

### Current validated live default
- **default live mode:** `balanced_overlay_mode`
- allocation: `{"pair_fast_exit": 0.2, "soft_three_way_regime": 0.8}`
- fallback: `risk_off_cash`
- pair: tactical-only

## Why hybrid was NOT promoted after the full rerun
Even after the heavy reboot rerun, hybrid is still much stronger on refreshed OOS than balanced:
- hybrid OOS: `+0.6868%`, Sharpe `3.2370`, max DD `0.2573%`
- balanced OOS: `+0.1091%`, Sharpe `0.4828`, max DD `0.5162%`

However, the current superiority gate still **does not promote hybrid to the live default**, because the gate also requires hybrid to avoid too much validation giveback versus balanced:
- hybrid val: `+6.5372%`, Sharpe `3.2857`
- balanced val: `+8.3078%`, Sharpe `4.1120`

So the final validated answer for this rerun is:
- **hybrid is the strongest diversified / guarded challenger**
- **balanced remains the live default**

## Current hybrid artifacts
- curated best now points to `config_name=long_memory`, but its refreshed OOS is tiny and it is not the chosen live config
- optuna `live_guarded` best trial: `9`
- optuna `train_aware_guarded` best trial: `9`
- final selected hybrid online config:
  - variant: `dynamic_default`
  - lookback_days: `25`
  - pair_weight_cap: `0.2467759186`
  - score_temperature: `1.0545357659`
  - sticky_default_bonus: `0.0466296276`
  - switch_margin: `0.0081550283`
- hybrid readiness:
  - `beats_cash_refreshed = true`
  - `beats_balanced_refreshed = true`
  - `beats_pair_tactical_refreshed = true`
  - `pair_cap_respected = true`
  - `recommended_stage = pilot_candidate`

## Canonical artifacts to read first next session
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/refreshed_operating_switch_current/portfolio_operating_switch_latest.md`
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/portfolio_hybrid_online_current/hybrid_online_portfolio_latest.json`
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/portfolio_candidate_overlay_review_current/portfolio_operating_playbook_latest.md`
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/final_master_scoreboard_current/portfolio_master_scoreboard_latest.md`
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/final_master_scoreboard_current/portfolio_operating_recommendation_onepager_latest.md`
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/final_master_scoreboard_current/hybrid_guarded_mode_runbook_latest.md`

## Important repo workflow note
`write_portfolio_operating_switch.py` now has a reproducible reboot-lane profile:

```bash
uv run python scripts/research/write_portfolio_operating_switch.py --artifact-profile reboot_validation
```

That profile freezes the market-state recompute to the reboot validation cutoff instead of drifting to newer live feature-point dates.

## Recommended next goal
If the goal is still to promote `hybrid_guarded_mode` in mixed/calm conditions, the next work should focus on **switch-threshold policy**, not rerun risk:
- decide whether the validation giveback guard is too strict
- or define a second policy lane where strong OOS + lower drawdown can outweigh weaker val
- keep pair tactical as override-only

