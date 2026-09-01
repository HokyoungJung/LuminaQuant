# Worker-1 clean_input_meta_selector current-promotion rejection

- Generated UTC: `2026-06-07T07:05:57Z`
- Worker: `worker-1`; Task: `4`
- Ultragoal evidence: `G002-contamination-and-eligibility-audit`
- Real-money approval blocked: **true**

## 100% threshold policy

The 100% annualized threshold is permitted only as a **post-evaluation reporting label**. It must not be a selector objective, Optuna objective, rerun trigger, or promotion filter.

## Decision

`clean_input_meta_selector` has historical locked-OOS annualized approx **110.46%** (OOS comp **85.91%**) but is **rejected for current promotion** and capped at `shadow_freeze_only` because the selector-grid ranking used historical locked-OOS context. It may remain a freeze/shadow benchmark only until fresh-forward evidence and paper fill telemetry exist.

## Required controls

| Artifact / candidate | Classification | Promotion decision | Metrics snapshot | Blockers |
| --- | --- | --- | --- | --- |
| `clean_input_meta_selector` | `shadow_freeze_only` | rejected | annualized_oos_return_approx_pct=110.46, hit_rate=5/10, max_bar_oos_mdd_pct=19.29, monthly_equity_mdd_pct=6.32, oos_compounded_return_pct=85.91 | post_oos_selector_grid_ranking_uses_historical_locked_oos; fresh_forward_required_before_promotion; paper_fill_telemetry_absent; 15bps_stress_artifact_absent |
| `strict_no_leak_best_single_10bps` | `eligible_control` | no_real_money; paper/control only | 10bps_mdd_pct=30.63, 10bps_pf=1.2096, 10bps_positive_months=6/10, 10bps_sharpe=1.26, 10bps_total_return_pct=54.56, 20bps_mdd_pct=43.63, 20bps_total_return_pct=27.1 | drawdown too high for real sleeve; 20bps stress MDD tail risk; 15bps/paper-fill telemetry missing; eligible universe only 10/85 symbols |
| `dynamic_conviction_switch_85_symbol_baseline` | `eligible_control` | paper baseline/monitor only | annualized_oos_return_approx_pct=42.57, cost_bps=10.0, hit_rate=3/10, max_oos_mdd_pct=27.69, oos_compounded_return_pct=34.39 | below 100pct reporting label; sparse positive folds; high MDD; no 15bps/20bps/paper fill telemetry in this artifact |

## Evidence paths

- `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_research_latest.json`
- `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_research_latest.md`
- `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_freeze_manifest_latest.json`

## Subagent evidence

Subagent skip reason: Task 4 reuses the completed Task 1 parallel repository-map and review probes covering the same clean_input_meta_selector risk surface; no additional serial repo-search/read loop was performed, so spawning new required probes would duplicate evidence.
