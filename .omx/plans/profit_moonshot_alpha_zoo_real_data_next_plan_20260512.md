# Profit Moonshot Next Plan — Alpha Zoo Real-Data Calibration (2026-05-12)

## What to use now

Use `CryptoFxAlphaZooStateStrategy` as the next primary research path.

Do **not** use the old current-base/calendar tuple as a live candidate or selection target. It remains a `hypothesis_reference_only` teacher/reference because its locked-OOS economics are strong but its calendar-primary/fixed month behavior is invalid for live promotion.

Use these existing components:

- Alpha Zoo factor layer: `src/lumina_quant/alpha_zoo/`
- Outcome/calibration layer: `src/lumina_quant/research/triple_barrier.py`, `candidate_outcome_ledger.py`, `edge_calibration.py`
- Strategy scaffold: `src/lumina_quant/strategies/crypto_fx_alpha_zoo_state.py`
- Research scripts:
  - `scripts/research/run_crypto_fx_alpha_zoo_screen.py`
  - `scripts/research/replay_crypto_fx_alpha_zoo_state.py`
  - `scripts/research/calibrate_crypto_fx_edges.py`
- State-distilled baseline/seed: `state_distilled_leadership_unwind` and `state_distilled_external_risk_filter`

## Current status

- Latest pushed green head: `fcc63f6c053c451152b0d780fa84ee91b5512f82` on `private/main`.
- GitHub Actions green:
  - private-ci: `https://github.com/hoky1227/Quants-agent/actions/runs/25738751636`
  - ci: `https://github.com/hoky1227/Quants-agent/actions/runs/25738751722`
- Calendar/current-base reference remains economically strong but invalid:
  - locked-OOS return `+6.4281%`
  - return/MDD `6.9169`
  - role: `hypothesis_reference_only`, not selector/live strategy
- Best valid state-distilled external-risk strict row at 4x:
  - `fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125`
  - train `+30.9030%`, validation `+12.4704%`, locked-OOS `+2.4852%`
  - liquidation count `0/0/0`, min margin buffers positive
  - strategy-valid, but `deployable_success=false` because it does not beat current-base reference economics
- Crypto/FX Alpha Zoo foundation exists but is only a smoke/research scaffold. It has not yet been run on real full current-tail economics.

## Non-negotiable constraints

1. No calendar/month/day/hour entry rules in valid candidate families.
2. Invalid current-base/calendar tuple is teacher/reference only; never a selection target.
3. Train/validation-only selection. Locked-OOS opens only after candidate freeze as gate/report-only.
4. Factor cards and strategy-validity metadata must fail closed on calendar-primary or OOS-selected candidates.
5. Strict deploy lane requires:
   - train/validation/OOS liquidation count `0`
   - every split min margin buffer `> 0`
   - OOS MDD `<= 25%`
   - OOS return and return/MDD beat baseline reference
6. Diagnostic nonfatal 5x/6x lane must remain separate and non-promotional.
7. Memory must stay below 8 GiB.

## Next session TODO

### A. Wire Alpha Zoo to real data

- Inspect current current-tail crypto panel/data loaders used by profit moonshot replays.
- Build a real-data input adapter for `run_crypto_fx_alpha_zoo_screen.py` instead of synthetic smoke data.
- Include crypto OHLCV/funding/OI/flow fields when available.
- Include FX/regime features if reliable data exists; otherwise use FRED lagged risk-state as a temporary regime filter and record that FX direct trading is still blocked.

### B. Real train/validation factor screen

- Run Alpha Zoo screen on real current-tail data.
- Persist factor cards with:
  - `calendar_primary=false`
  - `uses_locked_oos_for_selection=false`
  - source/data coverage metadata
  - train/validation IC/quantile/triple-barrier stats
- Do not rank by locked-OOS.

### C. Triple-barrier candidate ledger

- Convert strategy/factor entries into triple-barrier outcomes.
- Append candidate ledger rows with realized net PnL, stop/take/time exit, MAE/MFE, cost/funding/spread fields where available.
- Bucket by `strategy × side × asset × regime × volatility/factor_bucket`.

### D. Edge calibration

- Calibrate only on train/validation.
- Require positive lower-confidence edge.
- Use shrinkage for sparse buckets.
- Block/downsize excessive tail-loss buckets.

### E. Stateful replay and portfolio comparison

Replay a narrow, interpretable set:

1. `CryptoFxAlphaZooStateStrategy` real-data calibrated variant.
2. `state_distilled_external_risk_filter` best valid seed.
3. Residual-pair/state-distilled combo only if selected by train/validation.

Freeze candidates on train/validation, then open locked-OOS only for report/gate.

### F. Liquidation-aware validation

- Run integer leverage grid `1x..6x`.
- Report:
  - strict zero-liquidation best leverage
  - diagnostic nonfatal 5x/6x performance
  - liquidation count/event drawdown/equity loss/recovery separately
- Do not promote any row with strict-lane liquidation count `>0` or min margin buffer `<=0`.

### G. Verification and handoff

Run at minimum:

```bash
uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q
uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q src scripts tests
git diff --check
git diff --cached --check
```

Then write/update:

- `.omx/notepad.md`
- `.omx/plans/profit_moonshot_alpha_zoo_real_data_next_plan_20260512.md` or successor
- `docs/research_note_profit_moonshot_alpha_zoo_real_data_20260512.md` or successor
- `docs/session_handoff_*alpha_zoo*_20260512.md` or successor
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_*` or successor

## Stop condition

Stop only when one of these is true:

1. A valid strict deployable candidate beats the invalid calendar reference on OOS return and return/MDD with zero liquidations and positive min buffers; or
2. Real-data Alpha Zoo path is exhausted with reproducible artifacts showing why no live promotion is allowed, plus the next best research seed is documented.


## Research-note update requirement

Every future session must update the active research notes before final handoff. At minimum, update `.omx/notepad.md`, the active `.omx/plans/*` file, and `docs/research_note_profit_moonshot_alpha_zoo_real_data_20260512.md` or its successor. If the session changes the global source inventory or research chronology, regenerate/update `docs/profit_moonshot_research_history_20260510.md` plus the matching `var/reports/profit_moonshot_20260501/research_history/` artifacts, or explicitly state why regeneration was not required.

## 2026-05-13 — Real-data Alpha Zoo calibrated replay result

- Ran real current-tail Alpha Zoo screen against `/home/hoky/Quants-agent/LuminaQuant/var/cache/profit_moonshot_fresh_start/joined_panel_de62df511cec53df6ad39521.parquet` with lagged FRED context; direct FX OHLCV trading stayed blocked because current-tail cache contains crypto OHLCV only.
- Factor/card validity passed fail-closed gates: `calendar_primary=false`, `uses_locked_oos_for_selection=false`, strategy validity pass.
- Candidate outcome ledger: `45160` rows; train+validation `30494`; locked-OOS `14666`.
- Edge calibration physically filtered to train/validation: input `45160`, calibration `30494`, locked-OOS calibration `0`, excluded locked-OOS `14666`.
- Replay grid selected `alpha_zoo_conservative_exit` from `9` formulaic candidates using train/validation metrics only; locked-OOS remained hidden until candidate freeze.
- Strict zero-liquidation lane highest safe integer: `6.0x`, liquidation count `0`, min buffer `9049.125962`, OOS return `41.0967%`, OOS MDD `13.6667%`, return/MDD `3.007073`, Sharpe `2.143209`.
- Deployable success remains `false`: OOS return beats the invalid current-base reference at strict 6x, but return/MDD `3.007073` is below current-base reference `6.916878`.
- Diagnostic 5x/6x lane is non-promotional and separate: 5x/6x both zero liquidation in this approximate replay, promotion_allowed=false.
- Peak RSS `512.711` MiB (<8 GiB).
- Artifacts: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_real_data_summary_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_state_replay_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/edge_calibration_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/candidate_outcome_ledger_latest.jsonl`.
- Research history/source ledger not regenerated: No new external source class or global chronology/source-ledger change; reused existing current-tail cache and 20260512 lagged FRED external-state artifact, added only session-scoped Alpha Zoo artifacts.

## 2026-05-14 — Strict policy restoration result

- Restored hard Alpha Zoo promotion gate: OOS return and OOS return/MDD must both beat the invalid current-base/calendar reference.
- Regenerated real-data artifacts under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/` using `--strict-real-data` and explicit output paths.
- Screen rows `58,845`; factor count `63`; selected cards `20`.
- Candidate ledger `67,259` rows; train+validation `45,311`; locked-OOS `21,948`.
- Edge calibration records `45,311`; locked-OOS calibration `0`; calibrated edge keys `12`.
- Replay selected `alpha_zoo_conservative_exit` from `9` formulaic train/validation candidates.
- Strict 6x lane: OOS return `41.0967%`, MDD `13.6667%`, return/MDD `3.007073`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`, liquidation `0`, min buffer `9049.125962`.
- Decision: `deployable_success=false`; return/MDD `3.007073` fails current-base reference `6.916878` despite OOS return beating reference.
- Diagnostic 5x/6x lane remains non-promotional and separate.
- Peak RSS `626.7266 MiB`; source ledger not regenerated because no new global source family was introduced.
