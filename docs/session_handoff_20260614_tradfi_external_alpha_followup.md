# 2026-06-14 KST — TradFi external-alpha follow-up / no-clean-improvement gate

## Scope

- User intent: after the 110-asset TradFi/external-alpha walk-forward, push harder for US-equity-aware alpha while preserving clean walk-forward discipline and real-money safety gates.
- Anchor artifacts:
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/tradfi_external_alpha_search_20260613/wf_110_asset_external_v1/tradfi_external_alpha_wf_110_asset_external_v1.json|md`
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/tradfi_external_alpha_search_20260613/wf_110_asset_external_v1/tradfi_external_alpha_improvement_summary_latest.json|md`
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/tradfi_external_alpha_search_20260613/wf_110_asset_external_v1/live_readiness_preflight_latest.json`
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/tradfi_external_alpha_search_20260613/wf_110_asset_external_v1/real_money_path_preflight_latest.json|md`
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/tradfi_external_alpha_search_20260613/wf_110_asset_external_v1/tradfi_external_alpha_improvement_followup_latest.json|md`
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/tradfi_external_alpha_search_20260613/wf_110_asset_external_v1/tradfi_raw_leadlag_diagnostic_probe_latest.json`
- TradFi assumption: US-equity-linked symbols require US cash-session awareness, stale/holiday/session guardrails, and conservative execution-cost stress before any paper/live promotion.

## Bottom line

No additional clean/promotable performance improvement was found. The repo now records this explicitly in `tradfi_external_alpha_improvement_followup_latest.json|md`.

| Candidate / attempt | Status | OOS comp | Max OOS MDD | Hit folds | Decision |
|---|---|---:|---:|---:|---|
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | best clean baseline | `+34.39%` | `27.69%` | `3/10` | clean, but hard-stop not promotable |
| `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled` | best moonshot/freeze candidate | `+79.42%` | `27.69%` | `4/10` | post-OOS/fresh-forward only |
| fast row-level selector sweep | follow-up attempt | `-18.87%` | `26.62%` | `4/10` | not promotable |
| clean new-alpha discovery, TradFi core + leaders | follow-up attempt | `-8.54%` | `9.98%` | `2/6` | not promotable |
| raw TradFi lead-lag train/validation selector | diagnostic moonshot attempt | `-91.21%` | `87.07%` | `1/10` | selector failed, not promotable |

Operational decision remains fail-closed:

- Real-money execution: **blocked**.
- Paper trading start: **blocked**.
- Shadow trading start: **blocked**.
- Allowed use of the 79.42% moonshot: research report, freeze candidate, fresh-forward shadow only.
- Blocked use of the moonshot: current locked-OOS promotion, paper trading, real money.

## What changed in code

- `scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py`
  - Added TradFi/external-alpha families and reporting support from the 110-asset WF branch.
  - Added 10/15/20bps cost-stress report schema for rows.
  - Added external-source references and non-leaf labels for TradFi/session/regime candidates.
  - Fixed Python syntax around `_safe_float` exception handling.
- `scripts/research/write_tradfi_external_alpha_real_money_path.py`
  - New fail-closed real-money path/preflight writer.
  - Requires valid WF summary, WF provenance, live-readiness preflight, hard-stop promotion, external-family improvement, and execution telemetry before enabling any mode.
- `scripts/research/write_tradfi_external_alpha_improvement_followup.py`
  - New follow-up writer that summarizes post-summary improvement attempts and freezes the no-clean-improvement decision.
  - Records row-level selector, clean-new-alpha, and raw lead-lag diagnostic evidence with source hashes.
- Tests added/extended:
  - `tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py`
  - `tests/test_tradfi_external_alpha_real_money_path.py`
  - `tests/test_tradfi_external_alpha_improvement_followup.py`

## Why performance was not promoted

1. The best clean row is positive but not robust enough for the hard-stop gate:
   - It does not beat the earlier challenger threshold of `+53.38%` OOS comp.
   - It carries `27.69%` max OOS drawdown.
   - Only `3/10` locked-OOS folds are positive.
2. The best headline return remains non-clean:
   - The `+79.42%` lagged-router row is explicitly marked `post_oos_research_variant` and `requires_fresh_forward_shadow`.
3. The most aggressive TradFi lead-lag probe overfit badly:
   - Static post-hoc sorting produced a large upper-bound row (`+1324.50%`, MDD `54.27%`, `3/10` positive), but this is invalid promotion evidence.
   - The train/validation selector version collapsed to `-91.21%` with `87.07%` max drawdown.
4. The clean new-alpha discovery with TradFi core + BTC/ETH/SOL leaders selected many rows but had negative aggregate OOS (`-8.54%`).
5. The live-readiness path still blocks all modes because fresh-forward evidence and execution telemetry are absent.

## Frozen next path

Use this as the next safe research path, not as execution permission:

1. Freeze the `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled` manifest and source hashes.
2. Do not edit thresholds/families after freeze.
3. Observe genuinely new forward data only.
4. Run fresh-forward shadow with no selector changes.
5. Require survival under 10/15/20bps stress and no hidden non-clean labels.
6. Only after that, collect paper/testnet fill, cancel, partial-fill, slippage, and reconciliation telemetry.
7. Keep real money blocked until every preflight gate in `real_money_path_preflight_latest.json` passes.

## Verification

Commands run after the code/report updates:

```sh
.venv/bin/python -m py_compile \
  scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py \
  scripts/research/write_tradfi_external_alpha_real_money_path.py \
  scripts/research/write_tradfi_external_alpha_improvement_followup.py

.venv/bin/python -m pytest \
  tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py \
  tests/test_tradfi_external_alpha_real_money_path.py \
  tests/test_tradfi_external_alpha_improvement_followup.py

.venv/bin/ruff check \
  scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py \
  scripts/research/write_tradfi_external_alpha_real_money_path.py \
  scripts/research/write_tradfi_external_alpha_improvement_followup.py \
  tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py \
  tests/test_tradfi_external_alpha_real_money_path.py \
  tests/test_tradfi_external_alpha_improvement_followup.py
```

Results:

- `py_compile`: pass.
- targeted pytest: `59 passed`.
- ruff: `All checks passed!`.

## Do-not-repeat notes

- Do not promote post-hoc static lead-lag rows, even when headline return is very high.
- Do not use current locked OOS to tune thresholds or choose a final production row.
- Do not call the moonshot real-money ready; it is only a fresh-forward shadow candidate.
- Do not start paper/shadow/real until `real_money_path_preflight_latest.json` no longer has blocking gates.
